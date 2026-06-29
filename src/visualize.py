"""Visualise CLS-token attention maps for MAE ViT on Flowers-102."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
from src.model import get_device, load_mae_pretrained_vit

MAE_CHECKPOINT = "models/mae_pretrain_vit_base.pth"
PATCH_GRID = 14          # 224 / 16 = 14 patches per side
NUM_PATCHES = PATCH_GRID * PATCH_GRID   # 196


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

_EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Return a normalised (1, 3, 224, 224) tensor from a PIL image."""
    return _EVAL_TF(pil_img).unsqueeze(0)


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) normalised tensor to a (H, W, 3) uint8 array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (tensor * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def extract_mean_attention(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Return the head-averaged CLS attention map, upsampled to (224, 224).

    Strategy:
      1. Temporarily disable fused_attn on the last block so that the
         explicit attn = softmax(q @ k^T) path runs and attn_drop is called.
      2. Hook attn_drop to capture attention weights (shape B, H, N+1, N+1).
      3. Take the CLS row (index 0) over patch tokens (indices 1:).
      4. Average across the 12 heads → (196,) → reshape (14, 14).
      5. Upsample to (224, 224) with bilinear interpolation via PIL.
    """
    last_attn = model.blocks[-1].attn

    # Disable fused path so attention weights are materialised
    orig_fused = last_attn.fused_attn
    last_attn.fused_attn = False

    captured: dict[str, torch.Tensor] = {}

    def _hook(module: nn.Module, inp, out):
        # out: (B, num_heads, N+1, N+1) — softmax weights after dropout (≡ identity at eval)
        captured["attn"] = out.detach().cpu()

    handle = last_attn.attn_drop.register_forward_hook(_hook)

    model.eval()
    with torch.no_grad():
        model(x.to(device))

    handle.remove()
    last_attn.fused_attn = orig_fused

    attn = captured["attn"]           # (1, 12, 197, 197)
    # CLS token (row 0) attending to each patch token (cols 1:)
    cls_attn = attn[0, :, 0, 1:]     # (12, 196)
    mean_attn = cls_attn.mean(0)      # (196,)

    patch_map = mean_attn.reshape(PATCH_GRID, PATCH_GRID).numpy()
    # Normalise to [0, 1] for visualisation
    patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + 1e-8)

    # Upsample to 224×224 with bilinear via PIL
    pil_map = Image.fromarray((patch_map * 255).astype(np.uint8), mode="L")
    pil_map = pil_map.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
    return np.array(pil_map).astype(np.float32) / 255.0   # (224, 224) in [0, 1]


# ---------------------------------------------------------------------------
# Overlay composer
# ---------------------------------------------------------------------------

def _make_overlay(rgb: np.ndarray, attn: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a Jet heatmap over the RGB image.

    Args:
        rgb:  (H, W, 3) uint8.
        attn: (H, W) float in [0, 1].
        alpha: Heatmap opacity.

    Returns:
        (H, W, 3) uint8 composite.
    """
    heatmap = (cm.jet(attn)[:, :, :3] * 255).astype(np.uint8)   # (H, W, 3)
    overlay = (rgb * (1 - alpha) + heatmap * alpha).clip(0, 255).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# Main visualisation function
# ---------------------------------------------------------------------------

def visualize_attention(
    model: nn.Module,
    device: torch.device,
    data_dir: str | Path = "data/",
    n_images: int = 5,
    save_path: str | Path = "results/attention_maps.png",
    seed: int = 42,
) -> None:
    """Sample *n_images* from the test set and save a (n_images × 3) attention grid.

    Columns: Original | Attention map | Overlay
    """
    random.seed(seed)
    data_dir = Path(data_dir)

    # Load test split without transforms to keep PIL images for display
    test_ds = datasets.Flowers102(
        root=data_dir, split="test", download=True, transform=None
    )
    indices = random.sample(range(len(test_ds)), n_images)

    fig, axes = plt.subplots(
        n_images, 3,
        figsize=(9, 3 * n_images),
        gridspec_kw={"wspace": 0.04, "hspace": 0.12},
    )
    col_titles = ["Original", "Attention map", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, pad=6)

    for row, idx in enumerate(indices):
        pil_img, label = test_ds[idx]
        pil_img = pil_img.convert("RGB")

        # Resize to 224 for display (same crop as eval transform)
        display_img = pil_img.resize((256, 256), resample=Image.BILINEAR)
        w, h = display_img.size
        left, top = (w - IMAGE_SIZE) // 2, (h - IMAGE_SIZE) // 2
        display_img = display_img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        rgb = np.array(display_img)

        # Model input
        x = _to_tensor(display_img)
        attn_map = extract_mean_attention(model, x, device)   # (224, 224)

        # Heatmap as RGB for display
        heatmap_rgb = (cm.jet(attn_map)[:, :, :3] * 255).astype(np.uint8)
        overlay     = _make_overlay(rgb, attn_map)

        for col, img_data in enumerate([rgb, heatmap_rgb, overlay]):
            ax = axes[row, col]
            ax.imshow(img_data)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"#{idx}\nclass {label}", fontsize=8, rotation=0,
                              labelpad=40, va="center")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Attention grid saved → {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise MAE ViT attention maps on Flowers-102 test images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",  default=MAE_CHECKPOINT,
                        help="Model weights (.pth). Defaults to MAE pretrained.")
    parser.add_argument("--data-dir",    default="data/")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--n-images",    type=int, default=5)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    _device = get_device()
    print(f"Device: {_device}")

    _model = load_mae_pretrained_vit(args.checkpoint)
    _model = _model.to(_device)

    visualize_attention(
        model=_model,
        device=_device,
        data_dir=args.data_dir,
        n_images=args.n_images,
        save_path=Path(args.results_dir) / "attention_maps.png",
        seed=args.seed,
    )
