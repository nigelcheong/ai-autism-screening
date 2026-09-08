"""Frozen-backbone embedding extraction shared by every image experiment.

``transform`` is a parameter of :func:`extract_embeddings`, not a hardcoded
pipeline -- this is what lets the blurred and background experiments reuse
this function unchanged, swapping only the transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

_BACKBONES: Dict[str, Tuple] = {
    "resnet50": (models.resnet50, models.ResNet50_Weights, "IMAGENET1K_V2"),
}


def get_backbone_weights_name(backbone: str) -> str:
    """Return the weights enum member name used for a registered backbone.

    Args:
        backbone: Backbone key, e.g. ``"resnet50"``.

    Returns:
        The weights enum member name, e.g. ``"IMAGENET1K_V2"``.

    Raises:
        KeyError: If ``backbone`` is not registered.
    """
    if backbone not in _BACKBONES:
        raise KeyError(f"Unknown backbone '{backbone}'. Available: {list(_BACKBONES)}")
    return _BACKBONES[backbone][2]


class _ImageDataset(Dataset):
    """Opens each path and applies ``transform``, in a fixed given order."""

    def __init__(self, paths: Sequence[Path], transform: Callable) -> None:
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.paths[idx])
        return self.transform(image)


def _build_frozen_backbone(backbone: str, device: torch.device) -> torch.nn.Module:
    if backbone not in _BACKBONES:
        raise KeyError(f"Unknown backbone '{backbone}'. Available: {list(_BACKBONES)}")

    builder_fn, weights_cls, weights_name = _BACKBONES[backbone]
    weights = getattr(weights_cls, weights_name)
    model = builder_fn(weights=weights)
    model.fc = torch.nn.Identity()
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    model.to(device)
    return model


def extract_embeddings(
    paths: Sequence[Path],
    transform: Callable,
    backbone: str = "resnet50",
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """Extract frozen-backbone embeddings for a fixed, ordered list of images.

    Runs the backbone in ``eval()`` mode under ``no_grad``, reading the
    penultimate layer (2048-d for ``resnet50``, ``IMAGENET1K_V2`` weights).
    Never shuffles -- row ``i`` of the returned array corresponds to
    ``paths[i]``.

    Args:
        paths: Image file paths, in the exact order embeddings should be
            returned.
        transform: Callable mapping an opened PIL image to a model-ready
            tensor (see :func:`src.data.transforms.intact_transform`).
        backbone: Backbone key, looked up in an internal registry.
        batch_size: Batch size for extraction.
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        Float32 array of shape ``(len(paths), n_features)``.
    """
    torch_device = torch.device(device)
    model = _build_frozen_backbone(backbone, torch_device)
    dataset = _ImageDataset(paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(torch_device)
            features = model(batch)
            chunks.append(features.cpu().numpy().astype(np.float32))

    return np.concatenate(chunks, axis=0)


def save_embeddings(
    array: np.ndarray,
    paths: Sequence[Path],
    meta: Dict,
    out_dir: Path,
    name: str = "embeddings",
) -> None:
    """Persist an embedding array, its row-order path index, and metadata.

    Writes ``{name}.npy``, ``{name}_index.csv`` (the ``path`` column in
    array-row order, for join safety), and ``{name}_meta.json`` under
    ``out_dir``. ``name`` lets one output directory hold several cached
    arrays (e.g. one per blur sigma).

    Args:
        array: Embedding array, e.g. from :func:`extract_embeddings`.
        paths: Image paths in the same row order as ``array``.
        meta: Small JSON-serialisable dict recording provenance (backbone,
            weights version, transform name, torch version, device name,
            timestamp, ...).
        out_dir: Directory to write into (created if missing).
        name: Filename stem for the three written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{name}.npy", array)
    pd.DataFrame({"path": [str(p) for p in paths]}).to_csv(
        out_dir / f"{name}_index.csv", index=False
    )
    (out_dir / f"{name}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def load_embeddings(out_dir: Path, name: str = "embeddings") -> Tuple[np.ndarray, pd.DataFrame, Dict]:
    """Load a previously saved embedding array, its path index, and metadata.

    Args:
        out_dir: Directory previously written by :func:`save_embeddings`.
        name: Filename stem passed to the original :func:`save_embeddings`
            call.

    Returns:
        Tuple of ``(array, index, meta)``.
    """
    out_dir = Path(out_dir)
    array = np.load(out_dir / f"{name}.npy")
    index = pd.read_csv(out_dir / f"{name}_index.csv")
    meta = json.loads((out_dir / f"{name}_meta.json").read_text(encoding="utf-8"))
    return array, index, meta
