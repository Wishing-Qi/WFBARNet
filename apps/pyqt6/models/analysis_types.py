from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AnalysisAction:
    time_range: str
    label: str
    confidence: float
    detail: str
    start_frame: int = 0
    end_frame: int = 0
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass(slots=True)
class AnalysisResult:
    status: str = "success"
    actions: list[AnalysisAction] = field(default_factory=list)
    total_frames: int = 0
    fps: float = 0.0
    avg_confidence: float = 0.0
    valid_pose_frames: int = 0
    valid_track_frames: int = 0
    inference_seconds: float = 0.0
    video_path: str = ""
    output_dir: str = ""
    output_files: dict[str, str] = field(default_factory=dict)
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @classmethod
    def from_payload(cls, payload: Any) -> "AnalysisResult":
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict):
            actions: list[AnalysisAction] = []
            raw_actions = payload.get("actions", [])
            if isinstance(raw_actions, list):
                for item in raw_actions:
                    if isinstance(item, AnalysisAction):
                        actions.append(item)
                    elif isinstance(item, dict):
                        actions.append(
                            AnalysisAction(
                                time_range=str(item.get("time_range", "")),
                                label=str(item.get("label", "")),
                                confidence=float(item.get("confidence", 0.0)),
                                detail=str(item.get("detail", "")),
                                start_frame=int(item.get("start_frame", 0)),
                                end_frame=int(item.get("end_frame", 0)),
                                start_time=float(item.get("start_time", 0.0)),
                                end_time=float(item.get("end_time", 0.0)),
                            )
                        )
            return cls(
                status=str(payload.get("status", "success")),
                actions=actions,
                total_frames=int(payload.get("total_frames", 0)),
                fps=float(payload.get("fps", 0.0)),
                avg_confidence=float(payload.get("avg_confidence", 0.0)),
                valid_pose_frames=int(payload.get("valid_pose_frames", 0)),
                valid_track_frames=int(payload.get("valid_track_frames", 0)),
                inference_seconds=float(payload.get("inference_seconds", 0.0)),
                video_path=str(payload.get("video_path", "")),
                output_dir=str(payload.get("output_dir", "")),
                output_files=dict(payload.get("output_files", {})),
                message=str(payload.get("message", "")),
                payload=dict(payload),
            )
        return cls(status="error", message=str(payload), payload={"raw": payload})

    def summary(self) -> str:
        if self.message:
            return self.message
        return f"{self.status}, actions={self.action_count}"

    def to_display_text(self) -> str:
        lines = [
            f"状态: {self.status}",
            f"动作数: {self.action_count}",
            f"总帧数: {self.total_frames}",
            f"平均置信度: {self.avg_confidence * 100:.1f}%",
            f"有效姿态帧数: {self.valid_pose_frames}",
            f"有效轨迹帧数: {self.valid_track_frames}",
        ]
        if self.message:
            lines.append(f"说明: {self.message}")
        return "\n".join(lines)

