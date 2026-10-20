from datetime import datetime, timezone


def utc_iso(dt: datetime) -> str:
    """ISO 8601 string with an explicit UTC marker.

    Every `datetime.utcnow()`-stamped column in this app is naive (no tzinfo)
    but represents UTC. Plain `.isoformat()` on that omits any offset, and
    browsers parse an offset-less date-*time* string as local time (not
    UTC) -- so the client ends up displaying the raw UTC wall-clock value
    mislabeled as local, off by whatever the viewer's UTC offset is. Tagging
    it `Z` here makes `new Date(...)` on the frontend convert correctly.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")
