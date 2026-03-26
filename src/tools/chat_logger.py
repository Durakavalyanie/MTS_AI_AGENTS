from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


@dataclass
class TrajectoryLogger:
    log_file: Path
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.entries.append(
            {
                "ts": datetime.now(UTC).isoformat(),
                "event_type": event_type,
                "payload": payload,
            }
        )

    def flush(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.write_text(
            json.dumps(self.entries, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


def save_code_blocks(messages: list[dict[str, Any]], target_dir: Path) -> int:
    from src.chat_manager import parse_tool_calls
    
    target_dir.mkdir(parents=True, exist_ok=True)
    code_counter = 0
    code_index: list[dict[str, Any]] = []

    for msg_idx, message in enumerate(messages):
        content = str(message.get("content", ""))
        sender = str(message.get("name", "unknown"))
        
        calls = parse_tool_calls(content)
        for tool_name, args in calls:
            if tool_name in ("execute_code"):
                code = args.get("code")
                if code:
                    code_counter += 1
                    code_file = target_dir / f"code_{code_counter:04d}_{sender}.py"
                    code_file.write_text(code.strip() + "\n", encoding="utf-8")
                    code_index.append(
                        {
                            "code_file": code_file.name,
                            "message_index": msg_idx,
                            "sender": sender,
                            "tool": tool_name
                        }
                    )

    (target_dir / "index.json").write_text(
        json.dumps(code_index, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return code_counter
