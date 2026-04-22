from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AnalysisResult:
    status: str = "success"
    actions: int = 0
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "AnalysisResult":
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict):
            return cls(
                status=str(payload.get("status", "success")),
                actions=int(payload.get("actions", 0)),
                message=str(payload.get("message", "")),
                payload=dict(payload),
            )
        return cls(status="unknown", message=str(payload), payload={"raw": payload})

    def summary(self) -> str:
        if self.message:
            return self.message
        return f"{self.status}, actions={self.actions}"

    def to_display_text(self) -> str:
        details = [f"状态: {self.status}", f"动作数: {self.actions}"]
        if self.message:
            details.append(f"说明: {self.message}")
        return "\n".join(details)

