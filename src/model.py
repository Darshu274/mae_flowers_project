"""ViT-B/16 model factory with MAE pretrained weight loading."""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn

NUM_CLASSES = 102
_VIT_NAME = "vit_base_patch16_224"

# Keys present in MAE encoder checkpoints but absent from timm's ViT.
# (e.g. full MAE saves decoder weights too — all are prefixed "decoder_".)
_DECODER_PREFIX = "decoder"


def get_device() -> torch.device:
    """Return MPS → CUDA → CPU, whichever is available first."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def load_mae_pretrained_vit(
    checkpoint: str | Path,
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Return a ViT-B/16 initialised with MAE pretrained encoder weights.

    Key-mapping notes (confirmed by inspecting the checkpoint):
    - MAE encoder keys are identical to timm's ViT-B/16 keys.
    - The checkpoint has no 'head.*' keys (MAE is self-supervised, no
      classifier); timm's randomly-initialised head is kept as-is.
    - Full MAE checkpoints may include decoder keys (prefixed 'decoder_');
      these are silently dropped because they have no timm counterpart.
    - Loading is strict=False so missing head keys don't raise an error.

    Args:
        checkpoint: Path to mae_pretrain_vit_base.pth.
        num_classes: Size of the classification head.
        freeze_backbone: If True, freeze everything except head.weight/bias.
    """
    model = timm.create_model(_VIT_NAME, pretrained=False, num_classes=num_classes)
    _load_encoder_weights(model, Path(checkpoint))

    if freeze_backbone:
        _freeze_backbone(model)

    return model


def build_scratch_vit(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Return a randomly-initialised ViT-B/16 (no pretrained weights)."""
    return timm.create_model(_VIT_NAME, pretrained=False, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_trainable_params(model: nn.Module) -> dict[str, int]:
    """Return trainable and total parameter counts.

    Example::
        >>> info = count_trainable_params(model)
        >>> print(info)
        {'trainable': 768, 'total': 86_567_656, 'frozen': 85_798_888}
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total, "frozen": total - trainable}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_encoder_weights(model: nn.Module, checkpoint_path: Path) -> None:
    """Load MAE encoder weights into *model* in-place."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    mae_state: dict[str, torch.Tensor] = raw.get("model", raw)

    # Drop decoder weights that have no timm counterpart.
    encoder_state = {
        k: v for k, v in mae_state.items()
        if not k.startswith(_DECODER_PREFIX)
    }

    timm_state = model.state_dict()

    # Separate keys into three buckets for a clear diagnostic summary.
    loaded, skipped_shape, skipped_missing = [], [], []
    for key, weight in encoder_state.items():
        if key not in timm_state:
            skipped_missing.append(key)
        elif weight.shape != timm_state[key].shape:
            skipped_shape.append(f"{key}: MAE{tuple(weight.shape)} vs timm{tuple(timm_state[key].shape)}")
        else:
            timm_state[key] = weight
            loaded.append(key)

    # head.* keys are legitimately absent from the MAE checkpoint.
    head_keys = [k for k in timm_state if k.startswith("head")]

    model.load_state_dict(timm_state, strict=True)

    print(f"MAE weights loaded: {len(loaded)} encoder layers matched.")
    print(f"  Head keys kept (random init): {head_keys}")
    if skipped_shape:
        print(f"  Skipped (shape mismatch): {skipped_shape}")
    if skipped_missing:
        print(f"  Skipped (not in timm):    {skipped_missing}")


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classification head."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head")
