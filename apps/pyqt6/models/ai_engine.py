from __future__ import annotations

import time

from apps.pyqt6.models.data_types import AnalysisResult


class AIEngine:
    def __init__(self) -> None:
        self._last_video_path: str | None = None

    def process_video(self, video_path: str) -> AnalysisResult:
        self._last_video_path = video_path
        time.sleep(1.5)
        return AnalysisResult(
            status="success",
            actions=5,
            message=f"已完成对 {video_path} 的基础分析",
            payload={"video_path": video_path},
        )

