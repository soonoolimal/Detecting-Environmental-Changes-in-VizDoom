import torch
import torch.nn.functional as F
from torch import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

from trainer.base_trainer import BaseTrainer


class DTTrainer(BaseTrainer):
    """DecisionTransformer Trainer"""
    def __init__(
        self,
        model,
        device,
        scale_grad: bool,
        ac_loss_w: float,
        rtg_loss_w: float,
        **kwargs,
    ):
        super().__init__(model, device, ckpt_name="best_dt.pt", **kwargs)
        
        self._last_val_metrics = {"ac_ce": 0.0, "rtg_mse": 0.0, "ac_acc": 0.0}
        
        self.ac_loss_w = ac_loss_w
        self.rtg_loss_w = rtg_loss_w
        
        self.scale_grad = scale_grad
        self.scaler = GradScaler(self.device.type, enabled=scale_grad)
        
    def build_optimizer(self):
        return self.model.configure_optimizers(lr=self.lr, weight_decay=self.weight_decay)

    def _extra_ckpt(self) -> dict:
        return {"scaler_state_dict": self.scaler.state_dict()}

    def _compute_losses(self, batch):
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device).long()
        rewards = batch["rewards"].to(self.device)
        returns_to_go = batch["returns_to_go"].to(self.device)
        timesteps = batch["timesteps"].to(self.device)

        mask = batch.get("mask", None)
        if mask is None:
            mask = torch.ones(actions.shape, device=self.device, dtype=torch.long)
        else:
            mask = mask.to(self.device).long()
        valid = mask.bool()
        
        B, T = actions.shape
        
        with autocast(self.device.type, enabled=self.scale_grad):
            rtg_preds, ac_logits = self.model(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                mask=mask,
            )

            # Action Loss: CE (Discrete)
            logits_flat = ac_logits.reshape(B * T, -1)  # (BT,num_actions)
            actions_flat = actions.reshape(B * T)        # (BT,)
            valid_flat = valid.reshape(B * T)
            loss_ac = F.cross_entropy(logits_flat[valid_flat], actions_flat[valid_flat])

            # Return-To-Go Loss: MSE
            # predict R_{t+1} from h(a_t)
            if T >= 2:
                pred_rtg = rtg_preds[:, :-1, :]
                targ_rtg = returns_to_go[:, 1:, :].detach()
                rtg_std = targ_rtg.std().clamp(min=1.0)  # for normalization within batch
                pred_rtg = pred_rtg / rtg_std
                targ_rtg = targ_rtg / rtg_std
                valid_next = mask[:, 1:].bool()
                loss_rtg = (
                    (pred_rtg - targ_rtg).pow(2).sum(dim=-1)[valid_next].mean()
                )
            else:
                loss_rtg = torch.zeros((), device=self.device)

            loss = self.ac_loss_w * loss_ac + self.rtg_loss_w * loss_rtg

        return loss, loss_ac, loss_rtg, ac_logits, actions, valid

    def train_step(self, batch) -> torch.Tensor:
        loss, _, _, _, _, _ = self._compute_losses(batch)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.grad_clip is not None:
            if self.scale_grad:
                self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    def _eval(self, valid_loader, epoch: int) -> float:
        total_loss = total_ac = total_rtg = 0.0
        total_correct, total_valid_steps = 0, 0
        steps = 0
        with torch.no_grad():
            pbar = tqdm(valid_loader, total=len(valid_loader), leave=False, desc="[Eval DT]")
            for batch in pbar:
                loss, loss_ac, loss_rtg, ac_logits, actions, valid = self._compute_losses(batch)
                total_loss += float(loss)
                total_ac += float(loss_ac)
                total_rtg += float(loss_rtg)
                preds = ac_logits.argmax(dim=-1)  # (B,T)
                total_correct += int((preds[valid] == actions[valid]).sum())
                total_valid_steps += int(valid.sum())
                steps += 1

        n = max(steps, 1)
        acc = total_correct / max(total_valid_steps, 1) * 100
        self._last_val_metrics = {
            "ac_ce": total_ac / n,
            "rtg_mse": total_rtg / n,
            "ac_acc": acc,
        }
        print(
            f"val_ac_ce={self._last_val_metrics['ac_ce']:.4f}  "
            f"val_ac_acc={acc:.2f}%  "
            f"val_rtg_mse={self._last_val_metrics['rtg_mse']:.4f}"
        )
        return total_loss / n

    def _log_val(self, epoch: int, val_loss: float):
        if self.writer is None:
            return
        self.writer.add_scalar("DT/val_loss", val_loss, epoch)
        self.writer.add_scalar("DT/val_ac_ce", self._last_val_metrics["ac_ce"], epoch)
        self.writer.add_scalar("DT/val_ac_acc", self._last_val_metrics["ac_acc"], epoch)
        self.writer.add_scalar("DT/val_rtg_mse", self._last_val_metrics["rtg_mse"], epoch)