import hashlib


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def mask_api_key(last_four: str) -> str:
    """Short, readable mask for display, e.g. 'sk_••••••••vg-c'. The real key
    is never stored, so this is built from just the last 4 chars we kept."""
    return f"sk_{'•' * 8}{last_four}"
