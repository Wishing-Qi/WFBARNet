from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.models.mmpose_backend import MMPoseInferenceItem


@dataclass(slots=True)
class YoloPoseBackend:
    model_weight: str | None
    device: str
    conf_thr: float = 0.3
    max_persons: int = 2
    model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_weight:
            raise ValueError("YOLO pose backend requires model_weight, e.g. assets/weights/pose/yolo26s-pose.pt")
        if not Path(self.model_weight).exists():
            raise FileNotFoundError(f"YOLO pose weight file not found: {self.model_weight}")
        self.model = self._load_model()

    def _load_model(self) -> Any:
        self._configure_ultralytics()
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics is required for YOLO pose inference. Install it with `pip install ultralytics`."
            ) from exc
        self._patch_pose26_head()
        return YOLO(self.model_weight)

    def _configure_ultralytics(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        config_dir = project_root / ".ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    def _patch_pose26_head(self) -> None:
        try:
            from ultralytics.nn.modules import head  # type: ignore
            import torch
        except Exception:
            return
        if not hasattr(head, "Pose26") and hasattr(head, "Pose"):
            class Pose26(head.Pose):
                def forward(self, x):
                    bs = x[0].shape[0]
                    kpt_features = [self.cv4[i](x[i]) for i in range(self.nl)]
                    kpt = torch.cat(
                        [self.cv4_kpts[i](kpt_features[i]).view(bs, self.nk, -1) for i in range(self.nl)],
                        -1,
                    )
                    detections = head.Detect.forward(self, x)
                    if self.training:
                        return detections, kpt
                    pred_kpt = self.kpts_decode(bs, kpt)
                    if self.export:
                        return torch.cat([detections, pred_kpt], 1)
                    return torch.cat([detections[0], pred_kpt], 1), (detections[1], kpt)

            head.Pose26 = Pose26
        if not hasattr(head, "RealNVP"):
            class RealNVP(torch.nn.Module):
                def forward(self, x, *args, **kwargs):
                    return x

                def inverse(self, x, *args, **kwargs):
                    return x

            head.RealNVP = RealNVP

    def infer(self, image: np.ndarray) -> list[MMPoseInferenceItem]:
        result = self.model.predict(
            image,
            device=self._ultralytics_device(),
            conf=self.conf_thr,
            verbose=False,
        )[0]

        boxes = self._boxes(result)
        keypoints, scores = self._keypoints(result)
        if not boxes or not keypoints:
            return []

        items: list[MMPoseInferenceItem] = []
        order = sorted(range(len(boxes)), key=lambda idx: _box_area(boxes[idx]), reverse=True)
        for idx in order[: self.max_persons]:
            if idx >= len(keypoints):
                continue
            items.append(
                MMPoseInferenceItem(
                    bbox=boxes[idx],
                    keypoints=keypoints[idx],
                    scores=scores[idx],
                    coordinate_space="original",
                )
            )
        return items

    def _ultralytics_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        if self.device.startswith("cuda:"):
            return self.device.split(":", 1)[1]
        if self.device.startswith("cuda"):
            return "0"
        return self.device

    def _boxes(self, result: Any) -> list[list[float]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        return [[float(v) for v in row[:4]] for row in xyxy]

    def _keypoints(self, result: Any) -> tuple[list[list[list[float]]], list[list[float]]]:
        kpts = getattr(result, "keypoints", None)
        if kpts is None or getattr(kpts, "xy", None) is None:
            return [], []

        xy = kpts.xy.detach().cpu().numpy()
        conf = getattr(kpts, "conf", None)
        if conf is not None:
            score_array = conf.detach().cpu().numpy()
        else:
            score_array = np.ones(xy.shape[:2], dtype=np.float32)

        keypoints = [
            [[float(x), float(y)] for x, y in person]
            for person in xy
        ]
        scores = [
            [float(score) for score in person_scores]
            for person_scores in score_array
        ]
        return keypoints, scores


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
