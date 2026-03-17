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
        self._last_val_acc = 0.0

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

        return loss, logits, labels, mask

    def train_step(self, batch) -> torch.Tensor:
        loss, _, _, _ = self._compute_loss(batch)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None:
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss

    def _eval(self, valid_loader, epoch: int) -> float:
        total_loss = 0.0
        total_correct, total_valid_steps, steps = 0, 0, 0
        num_batches = len(valid_loader)
        with torch.no_grad():
            pbar = tqdm(valid_loader, total=num_batches, leave=False, desc="[Eval TSD]")
            for batch_idx, batch in enumerate(pbar):
                loss, logits, labels, mask = self._compute_loss(batch)
                total_loss += float(loss)
                steps += 1

                preds = logits.argmax(dim=-1)  # (B,T)
                valid = mask.bool()
                total_correct += int((preds[valid] == labels[valid]).sum())
                total_valid_steps += int(valid.sum())

                if self.writer is not None:
                    global_step = (epoch - 1) * num_batches + batch_idx

                    valid_preds = preds[valid]
                    valid_labels_batch = labels[valid]

                    # Per-batch predicted class distribution
                    self.writer.add_scalars("Td_val/pred_class_dist", {
                        "vanilla": int((valid_preds == 0).sum()),
                        "ob_shifted": int((valid_preds == 1).sum()),
                        "rew_shifted": int((valid_preds == 2).sum()),
                    }, global_step)

                    # Per-batch true class distribution
                    self.writer.add_scalars("Td_val/true_class_dist", {
                        "vanilla": int((valid_labels_batch == 0).sum()),
                        "ob_shifted": int((valid_labels_batch == 1).sum()),
                        "rew_shifted": int((valid_labels_batch == 2).sum()),
                    }, global_step)

                    # Per-batch shift flag
                    shift_flag = TaskDetector.detect_shift(logits, mask, omega=self.omega)
                    self.writer.add_scalar("Td_val/shift_flag", int(shift_flag), global_step)

        val_loss = total_loss / max(steps, 1)
        self._last_val_acc = total_correct / max(total_valid_steps, 1) * 100
        print(f"val_acc={self._last_val_acc:.2f}%")
        
        return val_loss

    def _log_val(self, epoch: int, val_loss: float):
        if self.writer is None:
            return
        self.writer.add_scalar("TD/val_loss", val_loss, epoch)
        self.writer.add_scalar("TD/val_acc", self._last_val_acc, epoch)

    def test(self, test_loader, num_classes: int = 3):
        """Test Loop
        
        Computes loss, overall accuracy, per-class accuracy, and confusion matrix.
        Logs results to TensorBoard and stdout.
        """
        self.model.eval()

        total_loss = 0.0
        total_correct, total_valid_steps, steps = 0, 0, 0
        # confusion[true_class][pred_class]
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

        with torch.no_grad():
            pbar = tqdm(test_loader, total=len(test_loader), leave=False, desc="[Test TSD]")
            for batch_idx, batch in enumerate(pbar):
                loss, logits, labels, mask = self._compute_loss(batch)
                total_loss += float(loss)
                steps += 1

                preds = logits.argmax(dim=-1)  # (B,T)
                valid = mask.bool()
                valid_preds = preds[valid].cpu()
                valid_labels = labels[valid].cpu()

                total_correct += int((valid_preds == valid_labels).sum())
                total_valid_steps += valid_labels.numel()

                # Confusion matrix accumulation
                for t in range(num_classes):
                    for p in range(num_classes):
                        confusion[t, p] += int(
                            ((valid_labels == t) & (valid_preds == p)).sum()
                        )

                # Per-batch logging
                if self.writer is not None:
                    # Per-batch predicted class distribution
                    self.writer.add_scalars("TD_Test/pred_class_dist", {
                        "vanilla": int((valid_preds == 0).sum()),
                        "ob_shifted": int((valid_preds == 1).sum()),
                        "rew_shifted": int((valid_preds == 2).sum()),
                    }, batch_idx)

                    # Per-batch true class distribution
                    self.writer.add_scalars("TD_Test/true_class_dist", {
                        "vanilla": int((valid_labels == 0).sum()),
                        "ob_shifted": int((valid_labels == 1).sum()),
                        "rew_shifted": int((valid_labels == 2).sum()),
                    }, batch_idx)

                    shift_flag = TaskDetector.detect_shift(logits, mask, omega=self.omega)
                    self.writer.add_scalar("TD_Test/shift_flag", int(shift_flag), batch_idx)

        n = max(steps, 1)
        test_loss = total_loss / n
        test_acc = total_correct / max(total_valid_steps, 1) * 100

        # Per-class accuracy from confusion matrix
        class_names = ["vanilla", "ob_shifted", "rew_shifted"]
        per_class_acc = {}
        for c in range(num_classes):
            total_c = confusion[c].sum().item()
            per_class_acc[class_names[c]] = (
                confusion[c, c].item() / total_c * 100 if total_c > 0 else float("nan")
            )

        # Stdout
        print(f"[Test] ACC: {test_acc:.2f}%  CE: {test_loss:.4f}")
        for name, acc in per_class_acc.items():
            print(f"  {name}: {acc:.2f}%")
        print("Confusion matrix (row=true, col=pred):")
        print(f"  {'':>12} " + "  ".join(f"{n:>12}" for n in class_names))
        for i, name in enumerate(class_names):
            row = "  ".join(f"{confusion[i, j].item():>12}" for j in range(num_classes))
            print(f"  {name:>12} {row}")

        # TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("TD_Test/loss", test_loss)
            self.writer.add_scalar("TD_Test/acc", test_acc)
            for name, acc in per_class_acc.items():
                self.writer.add_scalar(f"TD_Test/acc_{name}", acc)