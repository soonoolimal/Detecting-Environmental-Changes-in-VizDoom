import os
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from load import h5_dataset as h5ds
from load import torch_dataloader as tdl
from model import ObsAutoEncoder, ObsEncoder, DecisionTransformer, TaskDetector
from trainer import AETrainer, DTTrainer, TDTrainer

from config import (
    EncConfig,
    AELoadConfig, AEPretrainConfig,
    DTLoadConfig, GPT2Config, DTConfig, DTTrainConfig,
    TDLoadConfig, TDConfig, TDTrainConfig,
)


def debug(
    env_name: str,
    exp_name: str,
    gamma: float = 1.0,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    use_ae: bool = False,
    ae_pretrained: bool = False,
    dt_pretrained: bool = False,
    omega: float = 0.5,
):
    # Configs
    enc_cfg = EncConfig()
    gpt2_cfg = GPT2Config()
    dt_cfg = DTConfig()
    td_cfg = TDConfig()
    ae_load_cfg = AELoadConfig()
    dt_load_cfg = DTLoadConfig()
    td_load_cfg = TDLoadConfig()
    ae_train_cfg = AEPretrainConfig()
    dt_train_cfg = DTTrainConfig()
    td_train_cfg = TDTrainConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = os.path.join(os.getcwd(), "data", "datasets")
    model_dir = os.path.join(os.getcwd(), "model", "pretrained", f"{env_name}_{exp_name}")
    log_dir = os.path.join(os.getcwd(), "runs", f"{env_name}_{exp_name}")
    writer = SummaryWriter(log_dir=log_dir)

    # =============
    # Load Dataset
    # =============
    ds_van, ds_ob, ds_rew = h5ds.load_datasets(data_dir, env_name, exp_name, gamma)

    van_train, van_valid, van_test = h5ds.split_dataset(ds_van, train_ratio, valid_ratio)
    ob_train,  ob_valid,  ob_test  = h5ds.split_dataset(ds_ob,  train_ratio, valid_ratio)
    rew_train, rew_valid, rew_test = h5ds.split_dataset(ds_rew, train_ratio, valid_ratio)

    ds_train = h5ds.merge_dataset([van_train, ob_train, rew_train])
    ds_valid = h5ds.merge_dataset([van_valid, ob_valid, rew_valid])

    n_actions = ds_train.n_actions

    # ==============
    # Build Encoder
    # ==============
    enc = ObsEncoder(
        enc_dim=dt_cfg.hidden_size,
        in_channels=enc_cfg.in_channels,
        out_channels=enc_cfg.out_channels,
    )

    if use_ae:
        ae = ObsAutoEncoder(
            in_channels=enc_cfg.in_channels,
            feat_channels=enc_cfg.out_channels,
        )
        if ae_pretrained:
            ckpt = torch.load(os.path.join(model_dir, "best_ae.pt"), map_location=device)
            ae.load_state_dict(ckpt["model_state_dict"])
        else:
            ae_train_loader = tdl.make_ae_dataloader(ds_train.observations, **asdict(ae_load_cfg))
            ae_valid_loader = tdl.make_ae_dataloader(ds_valid.observations, **asdict(ae_load_cfg))
            ae_trainer = AETrainer(
                ae,
                device,
                env_name=env_name,
                exp_name=exp_name,
                denoise_std=ae_train_cfg.denoise_std,
                writer=writer,
                **{k: v for k, v in asdict(ae_train_cfg).items() if k not in ("denoise_std", "epochs")},
            )
            ae_trainer.pretrain(ae_train_loader, ae_valid_loader, epochs=ae_train_cfg.epochs)

        enc.load_backbone_from(ae.backbone)
        enc.freeze(only_backbone=True)

    # ===================================
    # Build & Train Decision Transformer
    # ===================================
    dt = DecisionTransformer(
        enc, n_actions,
        hidden_size=dt_cfg.hidden_size,
        seq_len=dt_cfg.seq_len,
        max_ep_len=dt_cfg.max_ep_len,
        **asdict(gpt2_cfg),
    )

    if dt_pretrained:
        ckpt = torch.load(os.path.join(model_dir, "best_dt.pt"), map_location=device)
        dt.load_state_dict(ckpt["model_state_dict"])
    else:
        dt_train_loader = tdl.make_dt_dataloader(ds_train, dt_cfg.seq_len, **asdict(dt_load_cfg))
        dt_valid_loader = tdl.make_dt_dataloader(ds_valid, dt_cfg.seq_len, **asdict(dt_load_cfg))
        dt_trainer = DTTrainer(
            dt, device,
            env_name=env_name,
            exp_name=exp_name,
            writer=writer,
            **{k: v for k, v in asdict(dt_train_cfg).items() if k != "epochs"},
        )
        dt_trainer.train(dt_train_loader, dt_valid_loader, epochs=dt_train_cfg.epochs)

    # ============================
    # Build & Train Task Detector
    # ============================
    td = TaskDetector(dt, ob_pred_dim=dt_cfg.hidden_size, **asdict(td_cfg))

    td_train_loader = tdl.make_td_dataloader([van_train, ob_train, rew_train], dt_cfg.seq_len, **asdict(td_load_cfg))
    td_valid_loader = tdl.make_td_dataloader([van_valid, ob_valid, rew_valid], dt_cfg.seq_len, **asdict(td_load_cfg))
    td_trainer = TDTrainer(
        td, device,
        env_name=env_name,
        exp_name=exp_name,
        writer=writer,
        **{k: v for k, v in asdict(td_train_cfg).items() if k != "epochs"},
    )
    td_trainer.train(td_train_loader, td_valid_loader, epochs=td_train_cfg.epochs)

    # ===================
    # Test Task Detector
    # ===================
    ckpt = torch.load(os.path.join(model_dir, "best_td.pt"), map_location=device)
    td.load_state_dict(ckpt["model_state_dict"])

    td_test_cfg = asdict(TDLoadConfig())
    td_test_cfg["shuffle"] = False
    td_test_loader = tdl.make_td_dataloader(
        [van_test, ob_test, rew_test], dt_cfg.seq_len, **td_test_cfg
    )

    td.eval()
    total_loss, total_correct, total_valid_steps, steps = 0.0, 0, 0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(td_test_loader):
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device).long()
            returns_to_go = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            mask = batch["mask"].to(device).long()
            labels = batch["labels"].to(device).long()

            logits = td(
                observations=observations,
                actions=actions,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                mask=mask,
            )  # (B,T,num_classes)

            valid = mask.bool()
            total_loss += float(F.cross_entropy(logits[valid], labels[valid]))
            preds = logits.argmax(dim=-1)  # (B,T)
            total_correct += int((preds[valid] == labels[valid]).sum())
            total_valid_steps += int(valid.sum())
            steps += 1

            # Per-Batch Logging
            # 1. Timestep-level class distribution (valid timesteps only)
            #    counts[c]: number of valid timesteps predicted as class c in this batch
            valid_preds = preds[valid]  # (num_valid_steps,)
            counts = {
                "vanilla": int((valid_preds == 0).sum()),
                "obs_shifted": int((valid_preds == 1).sum()),
                "rew_shifted": int((valid_preds == 2).sum()),
            }
            writer.add_scalars("TD_Test/pred_class_dist", counts, batch_idx)

            # 2. Shift flag for this batch (True if frac of non-vanilla >= omega)
            shift_flag = TaskDetector.detect_shift(logits, mask, omega=omega)
            writer.add_scalar("TD_Test/shift_flag", int(shift_flag), batch_idx)

    n = max(steps, 1)
    acc = total_correct / max(total_valid_steps, 1) * 100
    print(f"[Test] CE: {total_loss / n:.4f}  ACC: {acc:.2f}%")

    writer.add_scalar("TDTrainer/test_loss", total_loss / n)
    writer.add_scalar("TDTrainer/test_accuracy", acc)
    writer.close()


if __name__ == "__main__":
    debug(
        env_name="DefendLine",
        exp_name="default",
    )