import io
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
from PIL import Image

import torch


CLASS_NAMES = ["vanilla", "ob_shifted", "rew_shifted"]


def confusion_matrix_figure(confusion: torch.Tensor, class_names: list) -> matplotlib.figure.Figure:
    n = len(class_names)
    cm = confusion.numpy().astype(float)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums  # row-normalized (recall per class)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{int(cm[i, j])}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    
    return fig


def fig_to_tensor(fig: matplotlib.figure.Figure) -> torch.Tensor:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    plt.close(fig)
    arr = np.array(img) # (H,W,3)
    
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)