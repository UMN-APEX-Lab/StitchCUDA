from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any


class JsonlEventLog:
    """Append-only workflow event log for replay and debugging."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event_type: str, **payload: Any) -> None:
        event = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "type": event_type,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str, sort_keys=True) + "\n")
