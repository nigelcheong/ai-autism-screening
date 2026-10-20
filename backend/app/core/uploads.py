"""Local-disk storage for uploaded screening images (dev stand-in for
Supabase Storage). Files are saved under a UUID name so client-supplied
filenames never reach the filesystem path, and `resolve_image_path` re-checks
containment before any file is opened for inference.
"""

import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import settings

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def upload_root() -> Path:
    root = Path(settings.upload_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


_upload_root = upload_root  # internal alias, kept for brevity below


async def save_uploaded_image(file: UploadFile) -> str:
    """Validate and persist one uploaded image, returning its stored name."""
    ext = Path(file.filename or "").suffix.lower()
    if file.content_type not in ALLOWED_CONTENT_TYPES or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported image type: {file.filename}")

    max_bytes = settings.max_image_size_mb * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(400, f"Image too large (max {settings.max_image_size_mb}MB): {file.filename}")

    name = f"{uuid.uuid4().hex}{ext}"
    (_upload_root() / name).write_bytes(data)
    return name


def resolve_image_path(stored_name: str) -> Path:
    """Map a stored image name back to its file, rejecting path traversal."""
    root = _upload_root()
    candidate = (root / Path(stored_name).name).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise HTTPException(400, f"Unknown image: {stored_name}")
    return candidate


def _explanation_dir(kind: str) -> Path:
    d = _upload_root() / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def explanation_path(kind: str, stored_name: str) -> Path | None:
    """Server-derived filename only (never a client-supplied parameter), so
    this is a plain lookup rather than resolve_image_path's traversal guard.
    `kind` is "heatmaps" (Grad-CAM) or "shap_diagrams" (SHAP region diagram)."""
    candidate = _explanation_dir(kind) / Path(stored_name).name
    return candidate if candidate.is_file() else None


def heatmap_dir() -> Path:
    return _explanation_dir("heatmaps")


def heatmap_path(stored_name: str) -> Path | None:
    return explanation_path("heatmaps", stored_name)


def shap_diagram_dir() -> Path:
    return _explanation_dir("shap_diagrams")


def shap_diagram_path(stored_name: str) -> Path | None:
    return explanation_path("shap_diagrams", stored_name)
