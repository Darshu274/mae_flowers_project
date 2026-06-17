"""Evaluate a trained ViT checkpoint on the Flowers-102 test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import (
    build_scratch_vit,
    count_trainable_params,
    get_device,
    load_mae_pretrained_vit,
)

MAE_CHECKPOINT = "models/mae_pretrain_vit_base.pth"
NUM_CLASSES = 102


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full loader and return (logits, preds, labels) as numpy arrays."""
    model.eval()
    all_logits, all_preds, all_labels = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating", unit="batch"):
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels)

    return (
        torch.cat(all_logits).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def top1_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def top5_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    top5 = np.argsort(logits, axis=1)[:, -5:]          # (N, 5) highest scores
    hits = np.any(top5 == labels[:, None], axis=1)
    return float(hits.mean())


def per_class_accuracy(preds: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Return {class_id: accuracy} for every class present in *labels*."""
    accs: dict[int, float] = {}
    for cls in range(NUM_CLASSES):
        mask = labels == cls
        if mask.sum() == 0:
            accs[cls] = None          # class absent from this split
        else:
            accs[cls] = float((preds[mask] == cls).mean())
    return accs


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    normalize: bool = True,
) -> None:
    """Save a compact confusion-matrix heatmap.

    With 102 classes the matrix is too dense for per-cell text, so we render
    a colour-only heatmap and keep tick labels small.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_plot = cm.astype(float) / row_sums
        title = "Confusion Matrix (row-normalised)"
        fmt_label = "Accuracy per cell"
    else:
        cm_plot = cm
        title = "Confusion Matrix (counts)"
        fmt_label = "Count"

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, label=fmt_label)

    ticks = np.arange(0, NUM_CLASSES, 10)
    ax.set_xticks(ticks); ax.set_xticklabels(ticks, fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels(ticks, fontsize=7)
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    device = get_device()
    print(f"Device : {device}")

    # --- Build model skeleton ---
    if args.mode == "scratch":
        model = build_scratch_vit()
    else:
        freeze = args.mode == "linear_probe"
        model = load_mae_pretrained_vit(MAE_CHECKPOINT, freeze_backbone=freeze)

    # --- Load trained weights ---
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)

    param_info = count_trainable_params(model)
    print(f"Params : {param_info['total']:,} total\n")

    # --- Data ---
    _, _, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # --- Predictions ---
    logits, preds, labels = collect_predictions(model, test_loader, device)

    # --- Metrics ---
    top1 = top1_accuracy(preds, labels)
    top5 = top5_accuracy(logits, labels)
    per_class = per_class_accuracy(preds, labels)

    # Summarise per-class distribution
    valid_accs = [v for v in per_class.values() if v is not None]
    per_class_summary = {
        "mean":   round(float(np.mean(valid_accs)),  4),
        "median": round(float(np.median(valid_accs)), 4),
        "min":    round(float(np.min(valid_accs)),   4),
        "max":    round(float(np.max(valid_accs)),   4),
        "per_class": {str(k): (round(v, 4) if v is not None else None)
                      for k, v in per_class.items()},
    }

    results = {
        "checkpoint":      str(args.checkpoint),
        "mode":            args.mode,
        "top1_accuracy":   round(top1, 5),
        "top5_accuracy":   round(top5, 5),
        "per_class":       per_class_summary,
        "num_test_samples": int(len(labels)),
    }

    # --- Print summary ---
    print(f"\nTop-1 accuracy : {top1:.4f}  ({top1 * 100:.2f}%)")
    print(f"Top-5 accuracy : {top5:.4f}  ({top5 * 100:.2f}%)")
    print(f"Per-class mean : {per_class_summary['mean']:.4f}")
    print(f"Per-class min  : {per_class_summary['min']:.4f}  "
          f"max: {per_class_summary['max']:.4f}")

    # --- Confusion matrix ---
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, save_path=out_dir / "confusion_matrix.png", normalize=True)

    # --- Save JSON ---
    json_path = out_dir / "eval_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Metrics saved   → {json_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ViT-B/16 checkpoint on Flowers-102 test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a .pth file saved by train.py (e.g. models/best_linear_probe.pth)",
    )
    parser.add_argument(
        "--mode", choices=["linear_probe", "fine_tune", "scratch"], required=True,
        help="Must match the mode used during training (determines model architecture)",
    )
    parser.add_argument("--data-dir",    default="data/")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)

    evaluate_checkpoint(parser.parse_args())
