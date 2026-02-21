"""Thread-safe in-memory event ring buffer for the /river dashboard."""

import threading
from collections import deque
from datetime import datetime, timezone

_buffer = deque(maxlen=500)
_lock = threading.Lock()


def push(event_type: str, **kwargs):
    """Append an event to the ring buffer."""
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        **kwargs,
    }
    with _lock:
        _buffer.append(event)


def snapshot(limit: int = 100, event_type: str | None = None) -> list[dict]:
    """Return recent events, newest first. Optionally filter by type."""
    with _lock:
        items = list(_buffer)
    if event_type:
        items = [e for e in items if e["type"] == event_type]
    return list(reversed(items[-limit:]))
