from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any
import warnings

import numpy as np


COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def get_bone_pairs(skeleton_format: str = "coco") -> list[tuple[int, int]]:
    if skeleton_format != "coco":
        raise NotImplementedError(f"Unsupported skeleton format: {skeleton_format}")
    return [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
        (3, 5), (4, 6),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]


def create_bones(joints: np.ndarray, bone_pairs: list[tuple[int, int]] | None = None) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim < 4 or joints.shape[-2:] != (17, 2):
        raise ValueError(f"joints must end with shape (17, 2), got {joints.shape}")
    pairs = get_bone_pairs() if bone_pairs is None else bone_pairs
    bones = []
    for start, end in pairs:
        start_joint = joints[..., start, :]
        end_joint = joints[..., end, :]
        invalid = (np.all(start_joint == 0.0, axis=-1) | np.all(end_joint == 0.0, axis=-1))[..., None]
        bones.append(np.where(invalid, 0.0, end_joint - start_joint))
    return np.stack(bones, axis=-2).astype(np.float32)


def normalize_joints(
    joints_pixel: np.ndarray,
    bboxes: np.ndarray,
    center_align: bool = True,
) -> np.ndarray:
    joints = np.asarray(joints_pixel, dtype=np.float32)
    boxes = np.asarray(bboxes, dtype=np.float32)
    if joints.shape[-2:] != (17, 2):
        raise ValueError(f"joints_pixel must end with shape (17, 2), got {joints.shape}")
    if boxes.shape[-1] != 4:
        raise ValueError(f"bboxes must end with shape (4,), got {boxes.shape}")
    if joints.shape[:-2] != boxes.shape[:-1]:
        raise ValueError(f"joints and bboxes prefixes must match, got {joints.shape} and {boxes.shape}")

    width = boxes[..., 2] - boxes[..., 0]
    height = boxes[..., 3] - boxes[..., 1]
    dist = np.sqrt(width * width + height * height).astype(np.float32)
    dist = np.where(dist > 1e-6, dist, 1.0)

    x_raw = joints[..., 0]
    y_raw = joints[..., 1]
    x_norm = np.where(x_raw != 0.0, (x_raw - boxes[..., None, 0]) / dist[..., None], 0.0)
    y_norm = np.where(y_raw != 0.0, (y_raw - boxes[..., None, 1]) / dist[..., None], 0.0)

    if center_align:
        center_x = (boxes[..., 0] + boxes[..., 2]) * 0.5
        center_y = (boxes[..., 1] + boxes[..., 3]) * 0.5
        offset_x = (center_x - boxes[..., 0]) / dist
        offset_y = (center_y - boxes[..., 1]) / dist
        x_norm = np.where(x_raw != 0.0, x_norm - offset_x[..., None], 0.0)
        y_norm = np.where(y_raw != 0.0, y_norm - offset_y[..., None], 0.0)

    return np.stack((x_norm, y_norm), axis=-1).astype(np.float32)


def normalize_shuttlecock(shuttle_pixel: np.ndarray, video_width: int | float, video_height: int | float) -> np.ndarray:
    shuttle = np.asarray(shuttle_pixel, dtype=np.float32)
    if shuttle.shape[-1] != 2:
        raise ValueError(f"shuttle_pixel must end with shape (2,), got {shuttle.shape}")
    width = max(float(video_width), 1.0)
    height = max(float(video_height), 1.0)
    valid = np.isfinite(shuttle).all(axis=-1) & (shuttle[..., 0] >= 0.0) & (shuttle[..., 1] >= 0.0)
    out = np.zeros_like(shuttle, dtype=np.float32)
    out[..., 0] = np.where(valid, shuttle[..., 0] / width, 0.0)
    out[..., 1] = np.where(valid, shuttle[..., 1] / height, 0.0)
    return out


def project_points_by_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    homography = np.asarray(H, dtype=np.float64)
    if pts.shape[-1] != 2:
        raise ValueError(f"points must end with shape (2,), got {pts.shape}")
    if homography.shape != (3, 3):
        raise ValueError(f"H must have shape (3, 3), got {homography.shape}")
    flat = pts.reshape(-1, 2)
    homogeneous = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float64)], axis=1)
    projected = homogeneous @ homography.T
    denom = projected[:, 2:3]
    valid = np.abs(denom) > 1e-9
    xy = np.zeros((flat.shape[0], 2), dtype=np.float64)
    xy[valid[:, 0]] = projected[valid[:, 0], :2] / denom[valid[:, 0]]
    return xy.reshape(pts.shape).astype(np.float32)


def _court_bounds(court_info: dict[str, Any] | None) -> tuple[float, float, float, float]:
    info = court_info or {}
    return (
        float(info.get("border_L", info.get("left", 0.0))),
        float(info.get("border_R", info.get("right", 1.0))),
        float(info.get("border_U", info.get("top", 0.0))),
        float(info.get("border_D", info.get("bottom", 1.0))),
    )


def _scale_points_for_homography(
    points: np.ndarray,
    court_info: dict[str, Any] | None,
    video_width: int | float,
    video_height: int | float,
) -> np.ndarray:
    info = court_info or {}
    target_w = info.get("homography_width", info.get("scale_to_width"))
    target_h = info.get("homography_height", info.get("scale_to_height"))
    if target_w is None or target_h is None:
        return points
    scaled = np.asarray(points, dtype=np.float32).copy()
    scaled[..., 0] *= float(target_w) / max(float(video_width), 1.0)
    scaled[..., 1] *= float(target_h) / max(float(video_height), 1.0)
    return scaled


def normalize_position(
    feet_points: np.ndarray,
    H: np.ndarray | None,
    court_info: dict[str, Any] | None,
    video_width: int | float,
    video_height: int | float,
) -> np.ndarray:
    feet = np.asarray(feet_points, dtype=np.float32)
    if feet.shape[-2:] != (2, 2):
        raise ValueError(f"feet_points must end with shape (2, 2), got {feet.shape}")
    if H is None:
        warnings.warn(
            "BST player position fallback is using image-normalized feet midpoint because homography H is missing. "
            "BST_CG_AP accuracy will be degraded.",
            RuntimeWarning,
            stacklevel=2,
        )
        midpoint = feet.mean(axis=-2)
        return normalize_shuttlecock(midpoint, video_width, video_height)

    scaled_feet = _scale_points_for_homography(feet, court_info, video_width, video_height)
    feet_court = project_points_by_homography(scaled_feet, H)
    midpoint = feet_court.mean(axis=-2)
    border_l, border_r, border_u, border_d = _court_bounds(court_info)
    x_den = border_r - border_l
    y_den = border_d - border_u
    if abs(x_den) <= 1e-9 or abs(y_den) <= 1e-9:
        raise ValueError(f"Invalid court bounds: {(border_l, border_r, border_u, border_d)}")
    out = np.empty_like(midpoint, dtype=np.float32)
    out[..., 0] = (midpoint[..., 0] - border_l) / x_den
    out[..., 1] = (midpoint[..., 1] - border_u) / y_den
    return out


def sort_players_top_bottom(
    joints: np.ndarray,
    bboxes: np.ndarray,
    pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints_arr = np.asarray(joints)
    boxes_arr = np.asarray(bboxes)
    pos_arr = np.asarray(pos)
    if pos_arr.shape[-2:] != (2, 2):
        raise ValueError(f"pos must end with shape (2, 2), got {pos_arr.shape}")
    order = np.argsort(pos_arr[..., 1], axis=-1)
    if order.ndim != 1:
        raise ValueError("sort_players_top_bottom expects a single frame with exactly two players")
    return joints_arr[order], boxes_arr[order], pos_arr[order]


def make_seq_len_same(
    target_len: int,
    joints: np.ndarray,
    pos: np.ndarray,
    shuttle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    joints_arr = np.asarray(joints, dtype=np.float32)
    pos_arr = np.asarray(pos, dtype=np.float32)
    shuttle_arr = np.asarray(shuttle, dtype=np.float32)
    video_len = len(pos_arr)
    if len(joints_arr) != video_len or len(shuttle_arr) != video_len:
        raise ValueError("joints, pos, and shuttle must have the same temporal length")

    if video_len > target_len:
        need_padding = (video_len % target_len) > (target_len // 2)
        stride = video_len // target_len + int(need_padding)
        joints_arr = joints_arr[::stride][:target_len]
        pos_arr = pos_arr[::stride][:target_len]
        shuttle_arr = shuttle_arr[::stride][:target_len]
        new_video_len = len(pos_arr)
        if need_padding:
            pad_len = target_len - new_video_len
            joints_arr = np.pad(joints_arr, ((0, pad_len), *([(0, 0)] * (joints_arr.ndim - 1))))
            pos_arr = np.pad(pos_arr, ((0, pad_len), *([(0, 0)] * (pos_arr.ndim - 1))))
            shuttle_arr = np.pad(shuttle_arr, ((0, pad_len), (0, 0)))
    else:
        new_video_len = video_len
        pad_len = target_len - new_video_len
        joints_arr = np.pad(joints_arr, ((0, pad_len), *([(0, 0)] * (joints_arr.ndim - 1))))
        pos_arr = np.pad(pos_arr, ((0, pad_len), *([(0, 0)] * (pos_arr.ndim - 1))))
        shuttle_arr = np.pad(shuttle_arr, ((0, pad_len), (0, 0)))

    if len(joints_arr) != target_len or len(pos_arr) != target_len or len(shuttle_arr) != target_len:
        raise AssertionError("Failed to make BST sequence lengths equal")
    return joints_arr.astype(np.float32), pos_arr.astype(np.float32), shuttle_arr.astype(np.float32), int(new_video_len)


def build_jnb_bone(joints_norm: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints_norm, dtype=np.float32)
    bones = create_bones(joints, get_bone_pairs())
    return np.concatenate((joints, bones), axis=-2).astype(np.float32)


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    if is_dataclass(obj):
        return getattr(obj, key, default)
    return getattr(obj, key, default)


def _extract_frame_pose(frame_result: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
    poses = _value(frame_result, "pose", []) or []
    joints: list[np.ndarray] = []
    bboxes: list[np.ndarray] = []
    for person in poses:
        keypoints = np.asarray(_value(person, "keypoints", []), dtype=np.float32)
        bbox = np.asarray(_value(person, "bbox", []), dtype=np.float32)
        if keypoints.shape[0] < 17 or keypoints.shape[-1] < 2 or bbox.shape[0] < 4:
            continue
        joints.append(keypoints[:17, :2])
        bboxes.append(bbox[:4])
    return joints, bboxes


def _extract_shuttle(frame_result: Any) -> tuple[np.ndarray, bool]:
    track = _value(frame_result, "track", None)
    if track is None:
        return np.zeros(2, dtype=np.float32), False
    xy = np.asarray(_value(track, "ball_xy", [-1.0, -1.0]), dtype=np.float32)
    visible = bool(_value(track, "visible", 0))
    if xy.shape[0] < 2 or not visible or not np.isfinite(xy[:2]).all() or np.any(xy[:2] < 0):
        return np.zeros(2, dtype=np.float32), False
    return xy[:2], True


def _fallback_image_pos(joints: np.ndarray, bboxes: np.ndarray, video_width: int | float, video_height: int | float) -> np.ndarray:
    feet = joints[:, [15, 16], :]
    feet_valid = ~np.all(feet == 0.0, axis=-1)
    midpoint = np.zeros((len(joints), 2), dtype=np.float32)
    for idx in range(len(joints)):
        if feet_valid[idx].all():
            midpoint[idx] = feet[idx].mean(axis=0)
        else:
            bbox = bboxes[idx]
            midpoint[idx] = [(bbox[0] + bbox[2]) * 0.5, bbox[3]]
    return normalize_shuttlecock(midpoint, video_width, video_height)


def _in_court(pos: np.ndarray, eps: float = 0.01) -> np.ndarray:
    return (pos[:, 0] > -eps) & (pos[:, 0] < 1.0 + eps) & (pos[:, 1] > -eps) & (pos[:, 1] < 1.0 + eps)


def prepare_bst_sample(
    frames_or_detection_results: list[Any],
    video_width: int | float,
    video_height: int | float,
    H: np.ndarray | None,
    court_info: dict[str, Any] | None,
    seq_len: int,
) -> dict[str, Any]:
    raw_joints: list[np.ndarray] = []
    raw_bboxes: list[np.ndarray] = []
    raw_pos: list[np.ndarray] = []
    raw_shuttle: list[np.ndarray] = []
    failed_frames: list[int] = []
    warned_missing_h = False

    for frame_idx, frame_result in enumerate(frames_or_detection_results):
        joints_list, bbox_list = _extract_frame_pose(frame_result)
        shuttle_xy, shuttle_visible = _extract_shuttle(frame_result)
        failed = len(joints_list) < 2

        if not failed:
            joints_candidates = np.stack(joints_list).astype(np.float32)
            bbox_candidates = np.stack(bbox_list).astype(np.float32)
            if H is None:
                if not warned_missing_h:
                    warnings.warn(
                        "BST prepare_bst_sample is using image-normalized player positions because homography H is missing. "
                        "BST_CG_AP accuracy will be degraded.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_missing_h = True
                pos_candidates = _fallback_image_pos(joints_candidates, bbox_candidates, video_width, video_height)
                in_court_idx = np.arange(len(joints_candidates))
            else:
                feet = joints_candidates[:, [15, 16], :]
                feet_valid = ~np.any(np.all(feet == 0.0, axis=-1), axis=-1)
                pos_candidates = normalize_position(feet, H, court_info, video_width, video_height)
                in_court_idx = np.nonzero(_in_court(pos_candidates) & feet_valid)[0]
            failed = len(in_court_idx) != 2

        if failed:
            failed_frames.append(frame_idx)
            raw_joints.append(np.zeros((2, 17, 2), dtype=np.float32))
            raw_bboxes.append(np.zeros((2, 4), dtype=np.float32))
            raw_pos.append(np.zeros((2, 2), dtype=np.float32))
            raw_shuttle.append(np.zeros(2, dtype=np.float32))
            continue

        selected_joints = joints_candidates[in_court_idx]
        selected_bboxes = bbox_candidates[in_court_idx]
        selected_pos = pos_candidates[in_court_idx]
        selected_joints, selected_bboxes, selected_pos = sort_players_top_bottom(selected_joints, selected_bboxes, selected_pos)
        raw_joints.append(selected_joints.astype(np.float32))
        raw_bboxes.append(selected_bboxes.astype(np.float32))
        raw_pos.append(selected_pos.astype(np.float32))
        raw_shuttle.append(
            normalize_shuttlecock(shuttle_xy, video_width, video_height)
            if shuttle_visible
            else np.zeros(2, dtype=np.float32)
        )

    joints_pixel = np.stack(raw_joints).astype(np.float32)
    bboxes = np.stack(raw_bboxes).astype(np.float32)
    pos = np.stack(raw_pos).astype(np.float32)
    shuttle = np.stack(raw_shuttle).astype(np.float32)
    failed_mask = np.zeros((len(joints_pixel),), dtype=bool)
    failed_mask[failed_frames] = True
    joints_norm = normalize_joints(joints_pixel, bboxes, center_align=True)
    joints_norm[failed_mask] = 0.0
    pos[failed_mask] = 0.0
    shuttle[failed_mask] = 0.0
    joints_norm, pos, shuttle, video_len = make_seq_len_same(seq_len, joints_norm, pos, shuttle)
    return {
        "human_pose": build_jnb_bone(joints_norm),
        "joints": joints_norm,
        "pos": pos,
        "shuttle": shuttle,
        "video_len": np.asarray(video_len, dtype=np.int64),
        "failed_frames": np.asarray(failed_frames, dtype=np.int64),
    }


def prepare_bst_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must not be empty")
    human_pose = np.stack([np.asarray(sample["human_pose"], dtype=np.float32) for sample in samples])
    shuttle = np.stack([np.asarray(sample["shuttle"], dtype=np.float32) for sample in samples])
    pos = np.stack([np.asarray(sample["pos"], dtype=np.float32) for sample in samples])
    video_len = np.asarray([int(np.asarray(sample["video_len"]).item()) for sample in samples], dtype=np.int64)
    if human_pose.ndim != 5 or human_pose.shape[-2:] != (36, 2):
        raise ValueError(f"human_pose before flatten must have shape (B, T, 2, 36, 2), got {human_pose.shape}")
    if shuttle.shape[:2] != human_pose.shape[:2] or shuttle.shape[-1] != 2:
        raise ValueError(f"shuttle must have shape (B, T, 2), got {shuttle.shape}")
    if pos.shape[:3] != human_pose.shape[:3] or pos.shape[-1] != 2:
        raise ValueError(f"pos must have shape (B, T, 2, 2), got {pos.shape}")
    return {
        "human_pose": human_pose,
        "shuttle": shuttle,
        "pos": pos,
        "video_len": video_len,
        "samples": samples,
    }
