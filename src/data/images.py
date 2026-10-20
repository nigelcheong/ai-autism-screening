"""Manifest construction for the facial-image archive."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_CLASS_NORMALIZATION = {
    "autistic": "autistic",
    "non_autistic": "non_autistic",
    "non-autistic": "non_autistic",
}


def _normalize_class_label(folder_name: str) -> str:
    """Map an on-disk class folder name to a canonical label.

    Args:
        folder_name: Name of the class folder, e.g. ``"Non-Autistic"``.

    Returns:
        ``"autistic"`` or ``"non_autistic"``.
    """
    key = folder_name.strip().lower().replace("-", "_")
    return _CLASS_NORMALIZATION.get(key, key)


def build_image_manifest(data_root: Path) -> pd.DataFrame:
    """Build one row per image under ``data_root``.

    Globs two levels under ``data_root`` (split folder, then class folder).
    Class folder spelling is inconsistent in this dataset (``Non_Autistic``
    vs ``Non-Autistic``) -- normalised here to ``autistic`` / ``non_autistic``.
    Every level is sorted so row order is deterministic and reproducible
    across runs.

    Args:
        data_root: Root directory holding one folder per published split
            (``train``, ``test``, ``valid``), each holding one folder per
            class.

    Returns:
        DataFrame with columns ``path``, ``filename``, ``split``,
        ``class_label``, sorted by ``(split, class_label, filename)``.
    """
    data_root = Path(data_root)
    rows = []

    for split_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            class_label = _normalize_class_label(class_dir.name)
            for image_path in sorted(p for p in class_dir.iterdir() if p.is_file()):
                rows.append(
                    {
                        "path": str(image_path),
                        "filename": image_path.name,
                        "split": split_dir.name,
                        "class_label": class_label,
                    }
                )

    manifest = pd.DataFrame(rows, columns=["path", "filename", "split", "class_label"])
    return manifest.sort_values(["split", "class_label", "filename"]).reset_index(drop=True)
