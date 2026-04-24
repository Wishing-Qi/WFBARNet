from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.utils.structures import FrameResult, TrackResult


DEFAULT_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]
BALL_COLOR = (0, 0, 255)
TRAIL_COLOR = (0, 220, 255)


@dataclass
class TrackTrailRenderer:
    fps: float = 25.0
    history_seconds: float = 3.0
    current_radius: int = 8
    trail_radius: int = 4
    _points: deque[tuple[float, float, float, float]] = field(default_factory=deque)

    def draw(
        self,
        frame: np.ndarray,
        result: FrameResult,
        *,
        timestamp_ms: int | None = None,
    ) -> np.ndarray:
        canvas = frame.copy()
        _draw_pose(canvas, result)

        timestamp_s = self._timestamp_seconds(result.frame_id, timestamp_ms)
        if result.track.visible:
            x, y = map(float, result.track.ball_xy)
            self._points.append((timestamp_s, x, y, float(result.track.score)))
        self._prune(timestamp_s)
        self._draw_trail(canvas, timestamp_s)
        self._draw_current(canvas, result.track)
        return canvas

    def _timestamp_seconds(self, frame_id: int, timestamp_ms: int | None) -> float:
        if timestamp_ms is not None:
            return max(0.0, float(timestamp_ms) / 1000.0)
        fps = self.fps if self.fps > 0 else 25.0
        return max(0.0, float(frame_id) / fps)

    def _prune(self, timestamp_s: float) -> None:
        cutoff = timestamp_s - max(0.0, self.history_seconds)
        while self._points and self._points[0][0] < cutoff:
            self._points.popleft()

    def _draw_trail(self, canvas: np.ndarray, timestamp_s: float) -> None:
        for point_time, x, y, _ in self._points:
            age = max(0.0, timestamp_s - point_time)
            fade = max(0.15, 1.0 - age / max(self.history_seconds, 1e-6))
            radius = max(2, int(round(self.trail_radius + fade * 2)))
            thickness = 1 if fade < 0.55 else 2
            color = tuple(int(channel * fade) for channel in TRAIL_COLOR)
            cv2.circle(canvas, (int(round(x)), int(round(y))), radius, color, thickness)

    def _draw_current(self, canvas: np.ndarray, track: TrackResult) -> None:
        if not track.visible:
            return
        x, y = map(int, map(round, track.ball_xy))
        cv2.circle(canvas, (x, y), self.current_radius, BALL_COLOR, 2)
        cv2.circle(canvas, (x, y), self.current_radius + 6, TRAIL_COLOR, 1)
        cv2.putText(
            canvas,
            f"{track.score:.2f}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            BALL_COLOR,
            1,
        )


def _draw_pose(canvas: np.ndarray, result: FrameResult) -> None:
    for person in result.pose:
        x1, y1, x2, y2 = map(int, person.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 255), 2)
        for x, y in person.keypoints:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), -1)
        for a, b in DEFAULT_SKELETON:
            if a < len(person.keypoints) and b < len(person.keypoints):
                p1 = tuple(map(int, person.keypoints[a]))
                p2 = tuple(map(int, person.keypoints[b]))
                cv2.line(canvas, p1, p2, (255, 180, 0), 2)


def draw_result(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    canvas = frame.copy()
    _draw_pose(canvas, result)
    if result.track.visible:
        x, y = map(int, result.track.ball_xy)
        cv2.circle(canvas, (x, y), 8, BALL_COLOR, 2)
        cv2.putText(canvas, f"{result.track.score:.2f}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 1)
    return canvas


def save_visualization_video(frames: list[np.ndarray], results: list[FrameResult], path: Path, fps: float = 25.0) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail_renderer = TrackTrailRenderer(fps=fps)
    for frame, result in zip(frames, results):
        writer.write(trail_renderer.draw(frame, result))
    writer.release()
