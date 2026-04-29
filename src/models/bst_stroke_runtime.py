from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch import nn

from src.builders.bst_input_adapter import prepare_bst_batch, prepare_bst_sample
from src.models.bst_runtime import decode_merged_display_class, run_bst_inference
from src.utils.structures import FrameResult


DEFAULT_BST_COURT_INFO = {
    "border_L": 0.0,
    "border_R": 610.0,
    "border_U": 0.0,
    "border_D": 1340.0,
}


@dataclass
class BSTStrokeRecognizer:
    model: nn.Module
    device: str
    video_width: int
    video_height: int
    fps: float = 25.0
    court_info: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_BST_COURT_INFO))
    frame_buffer: deque[FrameResult] = field(init=False)

    def __post_init__(self) -> None:
        self.frame_buffer = deque(maxlen=max(1, self.seq_len))

    @property
    def seq_len(self) -> int:
        return int(getattr(self.model, "bst_seq_len", 30))

    def update(
        self,
        frame_result: FrameResult,
        *,
        hit_event: object | None,
        court_prediction: object | None,
    ) -> dict[str, Any] | None:
        self.frame_buffer.append(frame_result)
        if hit_event is None:
            return None

        event = _normalize_hit_event(hit_event, frame_result)
        h = extract_image_to_court_h(court_prediction)
        sample = prepare_bst_sample(
            list(self.frame_buffer),
            self.video_width,
            self.video_height,
            h,
            self.court_info,
            self.seq_len,
        )
        batch = prepare_bst_batch([sample])
        inference = run_bst_inference(
            self.model,
            batch["human_pose"],
            batch["shuttle"],
            batch["pos"],
            batch["video_len"],
            self.device,
        )
        top5_display = []
        if isinstance(inference.get("top5"), list):
            for item in inference["top5"]:
                if not isinstance(item, dict):
                    continue
                class_id = int(item.get("class_id", -1))
                top5_display.append(
                    {
                        **item,
                        "display_name": decode_merged_display_class(class_id)
                        if 0 <= class_id < 25
                        else str(class_id),
                    }
                )
        return {
            "event_frame_id": event["frame_id"],
            "timestamp_ms": event["timestamp_ms"],
            "hit_xy": event["ball_xy"],
            "pred_id": int(inference["pred_id"]),
            "pred_name": str(inference["pred_name"]),
            "pred_display_name": decode_merged_display_class(int(inference["pred_id"])),
            "confidence": float(inference["confidence"]),
            "top5": inference["top5"],
            "top5_display": top5_display,
            "video_len": int(batch["video_len"][0]),
            "seq_len": self.seq_len,
            "failed_frames": int(len(sample.get("failed_frames", []))),
            "used_homography": h is not None,
        }


def extract_image_to_court_h(court_prediction: object | None) -> np.ndarray | None:
    if court_prediction is None or not _prediction_value(court_prediction, "valid", False):
        return None
    raw_h = _prediction_value(court_prediction, "image_to_court_h", None)
    if raw_h is None:
        return None
    try:
        h = np.asarray(raw_h, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    return h


def _normalize_hit_event(hit_event: object, fallback_frame: FrameResult) -> dict[str, Any]:
    frame_id = int(_prediction_value(hit_event, "frame_id", fallback_frame.frame_id))
    timestamp_ms = int(_prediction_value(hit_event, "timestamp_ms", 0))
    ball_xy = _prediction_value(hit_event, "ball_xy", fallback_frame.track.ball_xy)
    try:
        x = float(ball_xy[0])  # type: ignore[index]
        y = float(ball_xy[1])  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        x, y = 0.0, 0.0
    return {
        "frame_id": frame_id,
        "timestamp_ms": max(0, timestamp_ms),
        "ball_xy": [x, y],
    }


def _prediction_value(prediction: object, key: str, default: object) -> object:
    if isinstance(prediction, dict):
        return prediction.get(key, default)
    return getattr(prediction, key, default)
