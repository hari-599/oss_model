import json
import time
import uuid
from pathlib import Path
from typing import Any
from guardrails import redact_sensitive_text


LOG_DIR = Path("logs")
EVENT_LOG = LOG_DIR / "events.jsonl"


def now_ms() -> int:
    return int(time.time() * 1000)


def new_trace_id() -> str:
    return str(uuid.uuid4())


def write_event(event: dict[str, Any]) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    safe_event = {
        key: redact_sensitive_text(value) if isinstance(value, str) else value
        for key, value in event.items()
    }
    with EVENT_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(safe_event, ensure_ascii=True) + "\n")
