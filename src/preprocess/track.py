from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch


@dataclass
class TrackPreprocessMeta:
    orig_size: tuple[int, int]
    resized_size: tuple[int, int]
    scale_x: float
    scale_y: float


def preprocess_track_window(
    frames: Sequence[np.ndarray],
    input_size: Tuple[int, int],
    device: str,
) -> tuple[torch.Tensor, TrackPreprocessMeta]:
    if len(frames) != 3:
        raise ValueError("Track branch expects exactly 3 frames.")
    in_w, in_h = input_size
    orig_h, orig_w = frames[1].shape[:2]
    resized = [cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR) for frame in frames]
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for frame in resized]
    stacked = np.concatenate(rgb, axis=2)
    tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).to(device)
    meta = TrackPreprocessMeta(
        orig_size=(orig_w, orig_h),
        resized_size=(in_w, in_h),
        scale_x=orig_w / float(in_w),
        scale_y=orig_h / float(in_h),
    )
    return tensor, meta


def preprocess_track_batch(
    batch_frames: list[Sequence[np.ndarray]],
    input_size: Tuple[int, int],
    device: str,
) -> tuple[torch.Tensor, list[TrackPreprocessMeta]]:
    if not batch_frames:
        raise ValueError("Batch is empty.")

    in_w, in_h = input_size
    batch_tensors = []

    metas = []
    for frames in batch_frames:
        if len(frames) != 3:
            raise ValueError("Each window must have exactly 3 frames.")

        orig_h, orig_w = frames[1].shape[:2]
        resized = [cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR) for frame in frames]
        rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for frame in resized]
        stacked = np.concatenate(rgb, axis=2)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1)
        batch_tensors.append(tensor)

        metas.append(TrackPreprocessMeta(
            orig_size=(orig_w, orig_h),
            resized_size=(in_w, in_h),
            scale_x=orig_w / float(in_w),
            scale_y=orig_h / float(in_h),
        ))

    batch_tensor = torch.stack(batch_tensors).to(device)
    return batch_tensor, metas
