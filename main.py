import os
import gc
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from load import h5_dataset as h5ds
from load import torch_dataloader as tdl
from model import ObsAutoEncoder, ObsEncoder, DecisionTransformer, TaskDetector
from trainer import AETrainer, DTTrainer, TDTrainer


def add_args():
    p = argparse.ArgumentParser()

    # General
    p.add_argument("-env", "--env_name", type=str, default="DefendLine")
    p.add_argument("-exp", "--exp_name", type=str, default="default")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--valid_ratio", type=float, default=0.2)
    p.add_argument("--omega", type=float, default=0.5)
    
    p.add_argument("--use_ae", action="store_true", default=False)
    p.add_argument("--ae_pretrained", action="store_true", default=False)
    p.add_argument("--dt_pretrained", action="store_true", default=False)

    # Encoder
    p.add_argument("--enc_in_channels", type=int, default=3)
    p.add_argument("--enc_out_channels", type=int, default=64)

    # Decision Transformer Architecture
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--max_ep_len", type=int, default=2100)
    p.add_argument("--seq_len", type=int, default=30)

    # GPT-2
    p.add_argument("--n_inner", type=int, default=None)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--resid_pdrop", type=float, default=0.1)
    p.add_argument("--embd_pdrop", type=float, default=0.1)
    p.add_argument("--attn_pdrop", type=float, default=0.1)
    p.add_argument("--layer_norm_epsilon", type=float, default=1e-5)

    # Task Detector Architecture
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=3)

    # AutoEncoder Dataloader
    p.add_argument("--ae_batch_size", type=int, default=128)
    p.add_argument("--ae_num_workers", type=int, default=4)

    # Decision Transformer Dataloader
    p.add_argument("--dt_batch_size", type=int, default=32)
    p.add_argument("--dt_num_workers", type=int, default=4)

    # Task Detector Dataloader
    p.add_argument("--td_batch_size", type=int, default=64)
    p.add_argument("--td_num_workers", type=int, default=4)

    # AutoEncoder Training
    p.add_argument("--ae_epochs", type=int, default=20)
    p.add_argument("--ae_lr", type=float, default=3e-4)
    p.add_argument("--ae_weight_decay", type=float, default=1e-4)
    p.add_argument("--ae_grad_clip", type=float, default=1.0)
    p.add_argument("--ae_patience", type=int, default=7)
    p.add_argument("--ae_denoise_std", type=float, default=0.05)

    # Decision Transformer training
    p.add_argument("--dt_epochs", type=int, default=30)
    p.add_argument("--dt_lr", type=float, default=3e-5)
    p.add_argument("--dt_weight_decay", type=float, default=1e-4)
    p.add_argument("--dt_grad_clip", type=float, default=1.0)
    p.add_argument("--dt_patience", type=int, default=10)
    p.add_argument("--scale_grad", action="store_true", default=False)
    p.add_argument("--ac_loss_w", type=float, default=0.5)
    p.add_argument("--rtg_loss_w", type=float, default=1.0)
    p.add_argument("--ob_loss_w", type=float, default=2.0)

    # Task Detector training
    p.add_argument("--td_epochs", type=int, default=20)
    p.add_argument("--td_lr", type=float, default=3e-4)
    p.add_argument("--td_weight_decay", type=float, default=1e-4)
    p.add_argument("--td_grad_clip", type=float, default=1.0)
    p.add_argument("--td_patience", type=int, default=7)

    return p


def main():
    args = add_args().parse_args()

    env_name = args.env_name
    exp_name = args.exp_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(os.getcwd(), "data", "datasets")
    model_dir = os.path.join(os.getcwd(), "model", "pretrained", f"{env_name}_{exp_name}")
    log_dir = os.path.join(os.getcwd(), "runs", f"{env_name}_{exp_name}")
    writer = SummaryWriter(log_dir=log_dir)

    # =============
    # Load Dataset
    # =============
    ds_van, ds_ob, ds_rew = h5ds.load_datasets(data_dir, env_name, exp_name, args.gamma)

    van_train, van_valid, van_test = h5ds.split_dataset(ds_van, args.train_ratio, args.valid_ratio)
    ob_train, ob_valid, ob_test = h5ds.split_dataset(ds_ob, args.train_ratio, args.valid_ratio)
    rew_train, rew_valid, rew_test = h5ds.split_dataset(ds_rew, args.train_ratio, args.valid_ratio)

    ds_train = h5ds.merge_dataset([van_train, ob_train, rew_train])
    ds_valid = h5ds.merge_dataset([van_valid, ob_valid, rew_valid])

    n_actions = ds_train.n_actions

    # ==============
    # Build Encoder
    # ==============
    enc = ObsEncoder(
        enc_dim=args.hidden_size,
        in_channels=args.enc_in_channels,
        out_channels=args.enc_out_channels,
    )

    if args.use_ae:
        ae = ObsAutoEncoder(
            in_channels=args.enc_in_channels,
            feat_channels=args.enc_out_channels,
        )
        if args.ae_pretrained:
            ckpt = torch.load(os.path.join(model_dir, "best_ae.pt"), map_location=device)
            ae.load_state_dict(ckpt["model_state_dict"])
        else:
            ae_train_loader = tdl.make_ae_dataloader(
                ds_train.observations,
                batch_size=args.ae_batch_size,
                num_workers=args.ae_num_workers,
            )
            ae_valid_loader = tdl.make_ae_dataloader(
                ds_valid.observations,
                batch_size=args.ae_batch_size,
                num_workers=args.ae_num_workers,
            )
            ae_trainer = AETrainer(
                ae, device,
                env_name=env_name,
                exp_name=exp_name,
                writer=writer,
                denoise_std=args.ae_denoise_std,
                lr=args.ae_lr,
                weight_decay=args.ae_weight_decay,
                grad_clip=args.ae_grad_clip,
                patience=args.ae_patience,
            )
            ae_trainer.pretrain(ae_train_loader, ae_valid_loader, epochs=args.ae_epochs)
            del ae_train_loader, ae_valid_loader, ae_trainer
            gc.collect()

        enc.load_backbone_from(ae.backbone)
        enc.freeze(only_backbone=True)
        del ae
        gc.collect()

    # ===================================
    # Build & Train Decision Transformer
    # ===================================
    dt = DecisionTransformer(
        encoder=enc,
        n_actions=n_actions,
        hidden_size=args.hidden_size,
        seq_len=args.seq_len,
        max_ep_len=args.max_ep_len,
        n_inner=args.n_inner,
        n_head=args.n_head,
        n_layer=args.n_layer,
        resid_pdrop=args.resid_pdrop,
        embd_pdrop=args.embd_pdrop,
        attn_pdrop=args.attn_pdrop,
        layer_norm_epsilon=args.layer_norm_epsilon,
    )

    if args.dt_pretrained:
        ckpt = torch.load(os.path.join(model_dir, "best_dt.pt"), map_location=device)
        dt.load_state_dict(ckpt["model_state_dict"])
    else:
        dt_train_loader = tdl.make_dt_dataloader(
            ds_train,
            seq_len=args.seq_len,
            batch_size=args.dt_batch_size,
            num_workers=args.dt_num_workers,
        )
        dt_valid_loader = tdl.make_dt_dataloader(
            ds_valid,
            seq_len=args.seq_len,
            batch_size=args.dt_batch_size,
            num_workers=args.dt_num_workers,
        )
        dt_trainer = DTTrainer(
            dt,
            device=device,
            env_name=env_name,
            exp_name=exp_name,
            writer=writer,
            lr=args.dt_lr,
            weight_decay=args.dt_weight_decay,
            grad_clip=args.dt_grad_clip,
            patience=args.dt_patience,
            scale_grad=args.scale_grad,
            ac_loss_w=args.ac_loss_w,
            rtg_loss_w=args.rtg_loss_w,
            ob_loss_w=args.ob_loss_w,
        )
        dt_trainer.train(dt_train_loader, dt_valid_loader, epochs=args.dt_epochs)
        
        del dt_train_loader, dt_valid_loader, dt_trainer
        gc.collect()
        torch.cuda.empty_cache()

    # ============================
    # Build & Train Task Detector
    # ============================
    td = TaskDetector(
        dt,
        ob_pred_dim=args.hidden_size,
        proj_dim=args.proj_dim,
        num_classes=args.num_classes,
    )

    td_train_loader = tdl.make_td_dataloader(
        datasets=[van_train, ob_train, rew_train],
        seq_len=args.seq_len,
        batch_size=args.td_batch_size,
        num_workers=args.td_num_workers,
    )
    td_valid_loader = tdl.make_td_dataloader(
        datasets=[van_valid, ob_valid, rew_valid],
        seq_len=args.seq_len,
        batch_size=args.td_batch_size,
        num_workers=args.td_num_workers,
    )
    td_trainer = TDTrainer(
        td,
        device=device,
        env_name=env_name,
        exp_name=exp_name,
        writer=writer,
        omega=args.omega,
        lr=args.td_lr,
        weight_decay=args.td_weight_decay,
        grad_clip=args.td_grad_clip,
        patience=args.td_patience,
    )
    td_trainer.train(td_train_loader, td_valid_loader, epochs=args.td_epochs)
    
    del td_train_loader, td_valid_loader
    gc.collect()

    # ===================
    # Test Task Detector
    # ===================
    ckpt = torch.load(os.path.join(model_dir, "best_td.pt"), map_location=device)
    td.load_state_dict(ckpt["model_state_dict"])

    td_test_loader = tdl.make_td_dataloader(
        datasets=[van_test, ob_test, rew_test],
        seq_len=args.seq_len,
        batch_size=args.td_batch_size,
        shuffle=False,
        num_workers=args.td_num_workers,
    )

    td_trainer.test(td_test_loader, num_classes=args.num_classes)
    writer.close()


if __name__ == "__main__":
    main()