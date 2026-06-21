"""Training script for MAE ViT on Oxford Flowers-102."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import get_dataloaders, get_few_shot_loader
from src.model import (
    build_scratch_vit,
    count_trainable_params,
    get_device,
    load_mae_pretrained_vit,
)

MAE_CHECKPOINT = "models/mae_pretrain_vit_base.pth"
EARLY_STOP_PATIENCE = 5


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    bar = tqdm(loader, desc=f"Epoch {epoch:>3d} [train]", leave=False, unit="batch")
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        n = images.size(0)
        total_loss += loss.item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        total += n
        bar.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "val",
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc=f"           [{desc}] ", leave=False, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        n = images.size(0)
        total_loss += loss.item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        total += n

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(mode: str) -> nn.Module:
    if mode == "scratch":
        return build_scratch_vit()
    freeze = mode == "linear_probe"
    return load_mae_pretrained_vit(MAE_CHECKPOINT, freeze_backbone=freeze)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"\nDevice : {device}")
    print(f"Mode   : {args.mode}")
    if args.few_shot_k:
        print(f"Few-shot: {args.few_shot_k} images/class")

    # --- Data ---
    _, val_loader, test_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    if args.few_shot_k:
        train_loader = get_few_shot_loader(
            args.data_dir, shots=args.few_shot_k,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
    else:
        train_loader, _, _ = get_dataloaders(
            args.data_dir, args.batch_size, args.num_workers
        )

    # --- Model ---
    model = _build_model(args.mode).to(device)
    param_info = count_trainable_params(model)
    print(
        f"Params : {param_info['trainable']:,} trainable  "
        f"/ {param_info['total']:,} total\n"
    )

    # --- Optimiser & scheduler ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)
    criterion = nn.CrossEntropyLoss()

    # --- Output paths ---
    run_name = args.mode
    if args.few_shot_k:
        run_name += f"_{args.few_shot_k}shot"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = Path(args.model_dir) / f"best_{run_name}.pth"
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    log: list[dict] = []
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = round(time.time() - t0, 1)
        lr_now = scheduler.get_last_lr()[0]

        entry = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  5),
            "val_loss":   round(val_loss,   5),
            "val_acc":    round(val_acc,    5),
            "lr":         round(lr_now,     8),
            "elapsed_s":  elapsed,
        }
        log.append(entry)
        (results_dir / "metrics.json").write_text(json.dumps(log, indent=2))

        # Best-model checkpoint
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            patience_counter = 0
        else:
            patience_counter += 1

        marker = " *" if improved else f"  (patience {patience_counter}/{EARLY_STOP_PATIENCE})"
        print(
            f"Epoch {epoch:>3d}/{args.epochs}"
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}"
            f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
            f"  lr={lr_now:.2e}  {elapsed}s{marker}"
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping: val_acc has not improved for {EARLY_STOP_PATIENCE} epochs.")
            break

    # --- Final test evaluation using best checkpoint ---
    print(f"\nLoading best checkpoint ({best_val_acc:.3f} val acc) for test evaluation...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="test")
    print(f"Test  loss={test_loss:.4f}  acc={test_acc:.4f}")

    summary = {
        "mode":         args.mode,
        "few_shot_k":   args.few_shot_k,
        "best_val_acc": round(best_val_acc, 5),
        "test_acc":     round(test_acc, 5),
        "epochs_run":   len(log),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {results_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ViT-B/16 on Oxford Flowers-102",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["linear_probe", "fine_tune", "scratch"],
        required=True,
        help="linear_probe: frozen backbone; fine_tune: all layers; scratch: no pretraining",
    )
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--weight-decay",type=float, default=0.05)
    parser.add_argument("--few-shot-k",  type=int,   default=None,
                        help="Images per class for few-shot training (e.g. 1, 2, 5)")
    parser.add_argument("--data-dir",    default="data/")
    parser.add_argument("--model-dir",   default="models/")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--num-workers", type=int,   default=4)

    main(parser.parse_args())
