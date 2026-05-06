"""Tiny helpers for emitting Server-Sent Events.

The SSE wire format is documented at
https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#fields

We use named events (``event: <name>``) so clients can dispatch on type, and
JSON payloads in the ``data:`` field. Each event is terminated with a blank
line (``\\n\\n``).
"""
from __future__ import annotations

import json
from typing import Any


def sse_event(event: str, data: dict[str, Any]) -> str:
    """Format one SSE frame: `event:` line + JSON `data:` line + terminator."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
