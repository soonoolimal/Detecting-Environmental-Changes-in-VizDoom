import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

from model.task_detector import TaskDetector
from trainer.base_trainer import BaseTrainer


class TDTrainer(BaseTrainer):
    """TaskDetector Trainer
    
    Only parameters with requires_grad=True are passed to the optimizer (DT backbone is frozen).
    """

    def __init__(
        self,
        model,
        device,
        omega: float = 0.5,
        **kwargs
    ):
        super().__init__(model, device, ckpt_name="best_td.pt", **kwargs)
        self.omega = omega

    def build_optimizer(self):
        return torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _compute_loss(self, batch):
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device).long()
        returns_to_go = batch["returns_to_go"].to(self.device)
        timesteps = batch["timesteps"].to(self.device)
        mask = batch["mask"].to(self.device).long()      # (B,T)
        labels = batch["labels"].to(self.device).long()  # (B,T)

        logits = self.model(
            returns_to_go=returns_to_go,
            observations=observations,
            actions=actions,
            timesteps=timesteps,
            mask=mask,
        )  # (B,T,num_classes)
        valid = mask.bool()
        loss = F.cross_entropy(logits[valid], labels[valid])

        return loss, logits, mask

    def train_step(self, batch) -> torch.Tensor:
        loss, _, _ = self._compute_loss(batch)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None:
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss

    def _eval(self, valid_loader, epoch: int) -> float:
        total_loss, steps = 0.0, 0
        num_batches = len(valid_loader)
        with torch.no_grad():
            pbar = tqdm(valid_loader, total=num_batches, leave=False, desc="[Eval TSD]")
            for batch_idx, batch in enumerate(pbar):
                loss, logits, mask = self._compute_loss(batch)
                total_loss += float(loss)
                steps += 1

                if self.writer is not None:
                    global_step = (epoch - 1) * num_batches + batch_idx

                    # Per-batch timestep-level class distribution
                    preds = logits.argmax(dim=-1)  # (B,T)
                    valid = mask.bool()
                    valid_preds = preds[valid]
                    counts = {
                        "vanilla":     int((valid_preds == 0).sum()),
                        "obs_shifted": int((valid_preds == 1).sum()),
                        "rew_shifted": int((valid_preds == 2).sum()),
                    }
                    self.writer.add_scalars("TD_Val/pred_class_dist", counts, global_step)

                    # Per-batch shift flag
                    shift_flag = TaskDetector.detect_shift(logits, mask, omega=self.omega)
                    self.writer.add_scalar("TD_Val/shift_flag", int(shift_flag), global_step)

        return total_loss / max(steps, 1)