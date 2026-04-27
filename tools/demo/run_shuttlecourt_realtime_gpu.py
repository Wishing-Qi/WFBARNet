from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SOURCE = PROJECT_ROOT / "videos" / "MVI_0212.MP4"
DEFAULT_WEIGHTS = PROJECT_ROOT / "assets" / "weights" / "ShuttleCourtNet" / "ShuttleCourt.pt"


@dataclass(slots=True)
class ShuttleCourtDetection:
    keypoints: np.ndarray
    scores: np.ndarray
    box_conf: float
    score: float


@dataclass(frozen=True, slots=True)
class CourtTemplate:
    width: float
    length: float
    corners: np.ndarray
    mask_polygon: np.ndarray
    lines: tuple[tuple[str, np.ndarray], ...]


@dataclass(slots=True)
class HomographyState:
    last_valid_h: np.ndarray | None = None
    last_valid_corners: np.ndarray | None = None
    last_reproj_error: float = float("nan")
    fallback_frames: int = 0


@dataclass(slots=True)
class CourtGeometryResult:
    h: np.ndarray | None
    raw_corners: np.ndarray | None
    smoothed_corners: np.ndarray | None
    valid: bool
    fallback: bool
    reason: str
    reproj_error: float = float("nan")
    mean_corner_conf: float = float("nan")
    corner_jump: float = float("nan")


COURT_WIDTH = 610.0
COURT_LENGTH = 1340.0
SINGLES_MARGIN = 46.0
NET_Y = COURT_LENGTH / 2.0
SHORT_SERVICE_FROM_NET = 198.0
DOUBLES_LONG_SERVICE_FROM_BACK = 76.0


def build_standard_court_template() -> CourtTemplate:
    """Return a full-size badminton court template in a local 2D plane.

    Coordinates are centimeters-like units:
    x=0..610 spans doubles sideline to doubles sideline,
    y=0..1340 spans far baseline to near baseline.
    """
    width = COURT_WIDTH
    length = COURT_LENGTH
    singles_left = SINGLES_MARGIN
    singles_right = width - SINGLES_MARGIN
    center_x = width / 2.0
    top_short_service = NET_Y - SHORT_SERVICE_FROM_NET
    bottom_short_service = NET_Y + SHORT_SERVICE_FROM_NET
    top_doubles_long_service = DOUBLES_LONG_SERVICE_FROM_BACK
    bottom_doubles_long_service = length - DOUBLES_LONG_SERVICE_FROM_BACK

    corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, length], [0.0, length]],
        dtype=np.float32,
    )
    lines: tuple[tuple[str, np.ndarray], ...] = (
        ("doubles_outer", corners),
        ("singles_left_sideline", np.array([[singles_left, 0.0], [singles_left, length]], dtype=np.float32)),
        ("singles_right_sideline", np.array([[singles_right, 0.0], [singles_right, length]], dtype=np.float32)),
        ("top_short_service", np.array([[0.0, top_short_service], [width, top_short_service]], dtype=np.float32)),
        ("bottom_short_service", np.array([[0.0, bottom_short_service], [width, bottom_short_service]], dtype=np.float32)),
        (
            "top_doubles_long_service",
            np.array([[0.0, top_doubles_long_service], [width, top_doubles_long_service]], dtype=np.float32),
        ),
        (
            "bottom_doubles_long_service",
            np.array([[0.0, bottom_doubles_long_service], [width, bottom_doubles_long_service]], dtype=np.float32),
        ),
        ("top_center_service", np.array([[center_x, 0.0], [center_x, top_short_service]], dtype=np.float32)),
        (
            "bottom_center_service",
            np.array([[center_x, bottom_short_service], [center_x, length]], dtype=np.float32),
        ),
    )
    return CourtTemplate(width=width, length=length, corners=corners, mask_polygon=corners, lines=lines)


STANDARD_COURT_TEMPLATE = build_standard_court_template()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime ShuttleCourt keypoint inference for videos/MVI_0212.MP4."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Video path or camera index.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO pose weights for ShuttleCourt.")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, cuda:0, ...")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO object confidence threshold.")
    parser.add_argument("--kp-conf", type=float, default=0.15, help="Low-confidence keypoints are drawn hollow.")
    parser.add_argument(
        "--corner-indices",
        "--court-indices",
        dest="corner_indices",
        default="0,1,2,3",
        help="Four outer-corner keypoint indices. They are reordered to TL,TR,BR,BL automatically.",
    )
    parser.add_argument("--net-indices", default="4,5,6,7", help="Four net keypoint indices.")
    parser.add_argument(
        "--surface-indices",
        default="0,2,3,1",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--template-mode",
        default="standard",
        choices=("standard",),
        help="Court template definition used for homography overlay.",
    )
    parser.add_argument(
        "--no-homography-overlay",
        dest="use_homography_overlay",
        action="store_false",
        help="Disable template projection and show only raw keypoints/net points.",
    )
    parser.set_defaults(use_homography_overlay=True)
    parser.add_argument(
        "--homography-kp-conf",
        type=float,
        default=0.10,
        help="Minimum confidence for the four corner keypoints used by homography.",
    )
    parser.add_argument(
        "--ransac-reproj-threshold",
        type=float,
        default=6.0,
        help="RANSAC reprojection threshold in pixels for cv2.findHomography.",
    )
    parser.add_argument(
        "--max-reproj-error",
        type=float,
        default=24.0,
        help="Reject a homography if mean corner reprojection error exceeds this many pixels.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.28,
        help="EMA weight for current-frame corner points. Higher follows motion faster.",
    )
    parser.add_argument(
        "--max-corner-jump-ratio",
        type=float,
        default=0.08,
        help="Reject sudden corner jumps above this ratio of the frame diagonal.",
    )
    parser.add_argument(
        "--max-fallback-frames",
        type=int,
        default=45,
        help="Reuse the last valid homography for this many consecutive invalid frames.",
    )
    parser.add_argument(
        "--min-court-area-ratio",
        type=float,
        default=0.005,
        help="Reject outer-court quadrilaterals smaller than this fraction of the frame area.",
    )
    parser.add_argument("--mask-alpha", type=float, default=0.16, help="Court mask opacity. Use 0 to disable.")
    parser.add_argument("--line-thickness", type=int, default=3)
    parser.add_argument("--point-radius", type=int, default=6)
    parser.add_argument("--show-labels", action="store_true", help="Draw raw keypoint numbers.")
    parser.add_argument("--save-video", default="", help="Optional output mp4 path.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means all frames.")
    parser.add_argument("--log-every", type=int, default=30, help="Print progress every N frames.")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without an OpenCV window. Useful on headless machines.",
    )
    parser.add_argument(
        "--realtime-playback",
        action="store_true",
        help="Sleep between frames to match the source FPS.",
    )
    return parser.parse_args()


def parse_indices(raw: str, expected: int | None = None) -> list[int]:
    indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if expected is not None and len(indices) != expected:
        raise ValueError(f"Expected {expected} indices, got {indices}.")
    return indices


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def configure_ultralytics() -> None:
    config_dir = PROJECT_ROOT / ".ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))


def patch_pose26_head() -> None:
    try:
        import torch
        from ultralytics.nn.modules import block
        from ultralytics.nn.modules import head
    except Exception:
        return

    real_nvp_class = None
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
        real_nvp_class = RealNVP
    else:
        real_nvp_class = head.RealNVP

    if not hasattr(block, "RealNVP"):
        block.RealNVP = real_nvp_class


def load_model(weights: str) -> Any:
    weight_path = Path(weights)
    if not weight_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    configure_ultralytics()
    patch_pose26_head()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install ultralytics") from exc

    return YOLO(str(weight_path))


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def select_best_detection(result: Any) -> ShuttleCourtDetection | None:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return None

    xy = to_numpy(getattr(keypoints, "xy", None))
    if xy.size == 0 or xy.ndim != 3:
        return None

    conf = to_numpy(getattr(keypoints, "conf", None))
    if conf.size == 0:
        conf = np.ones(xy.shape[:2], dtype=np.float32)

    box_conf = np.ones((xy.shape[0],), dtype=np.float32)
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        raw_box_conf = to_numpy(getattr(boxes, "conf", None))
        if raw_box_conf.size == xy.shape[0]:
            box_conf = raw_box_conf.astype(np.float32)

    best_index = -1
    best_score = -1.0
    for det_index in range(xy.shape[0]):
        kp_score = float(np.nanmean(conf[det_index]))
        score = kp_score * float(box_conf[det_index])
        if score > best_score:
            best_index = det_index
            best_score = score

    if best_index < 0:
        return None

    return ShuttleCourtDetection(
        keypoints=xy[best_index].astype(np.float32),
        scores=conf[best_index].astype(np.float32),
        box_conf=float(box_conf[best_index]),
        score=float(best_score),
    )


def infer_frame(model: Any, frame: np.ndarray, args: argparse.Namespace, device: str) -> ShuttleCourtDetection | None:
    results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)
    if not results:
        return None
    return select_best_detection(results[0])


def valid_points(points: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    height, width = frame_shape[:2]
    finite = np.isfinite(points).all(axis=1)
    in_x = (points[:, 0] >= -width * 0.05) & (points[:, 0] <= width * 1.05)
    in_y = (points[:, 1] >= -height * 0.05) & (points[:, 1] <= height * 1.05)
    return finite & in_x & in_y


def polygon_area(points: np.ndarray) -> float:
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return float(cv2.contourArea(contour))


def is_convex_quad(points: np.ndarray) -> bool:
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return bool(cv2.isContourConvex(contour))


def order_court_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder four arbitrary corner points to TL, TR, BR, BL in image space."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("corner points must be a finite (4, 2) array")

    y_sorted = np.argsort(pts[:, 1])
    top = y_sorted[:2]
    bottom = y_sorted[2:]
    top = top[np.argsort(pts[top, 0])]
    bottom = bottom[np.argsort(pts[bottom, 0])]
    order = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.int32)
    ordered = pts[order]

    if polygon_area(ordered) < 1.0 or not is_convex_quad(ordered):
        sums = pts[:, 0] + pts[:, 1]
        diffs = pts[:, 1] - pts[:, 0]
        fallback_order = np.array(
            [
                int(np.argmin(sums)),
                int(np.argmin(diffs)),
                int(np.argmax(sums)),
                int(np.argmax(diffs)),
            ],
            dtype=np.int32,
        )
        if len(set(int(index) for index in fallback_order)) != 4:
            raise ValueError("corner ordering is ambiguous")
        ordered = pts[fallback_order]
        order = fallback_order

    if polygon_area(ordered) < 1.0 or not is_convex_quad(ordered):
        raise ValueError("ordered corners do not form a valid convex quadrilateral")
    return ordered.astype(np.float32), order


def is_reasonable_court_quad(points: np.ndarray, frame_shape: tuple[int, ...], min_area_ratio: float) -> bool:
    height, width = frame_shape[:2]
    frame_area = float(max(width * height, 1))
    area = polygon_area(points)
    if area < frame_area * max(0.0, float(min_area_ratio)):
        return False
    if not is_convex_quad(points):
        return False
    side_lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    frame_diag = float(np.hypot(width, height))
    return bool(np.min(side_lengths) >= max(8.0, frame_diag * 0.006))


def project_points(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, h).reshape(-1, 2)


def reprojection_error(h: np.ndarray, image_points: np.ndarray) -> float:
    projected = project_points(STANDARD_COURT_TEMPLATE.corners, h)
    errors = np.linalg.norm(projected - np.asarray(image_points, dtype=np.float32), axis=1)
    return float(np.mean(errors))


def estimate_homography(
    image_points: np.ndarray,
    ransac_threshold: float = 6.0,
) -> tuple[np.ndarray | None, float, np.ndarray | None]:
    """Estimate template-plane to image homography from ordered TL,TR,BR,BL corners."""
    ordered = np.asarray(image_points, dtype=np.float32)
    if ordered.shape != (4, 2) or not np.isfinite(ordered).all():
        return None, float("inf"), None

    h, inliers = cv2.findHomography(
        STANDARD_COURT_TEMPLATE.corners,
        ordered,
        cv2.RANSAC,
        max(1.0, float(ransac_threshold)),
    )
    if h is None or h.shape != (3, 3) or not np.isfinite(h).all():
        return None, float("inf"), inliers
    if abs(float(h[2, 2])) > 1e-9:
        h = h / float(h[2, 2])
    return h.astype(np.float64), reprojection_error(h, ordered), inliers


def project_court_template(h: np.ndarray, template: CourtTemplate = STANDARD_COURT_TEMPLATE) -> dict[str, np.ndarray]:
    return {name: project_points(line, h) for name, line in template.lines}


def extract_ordered_corner_points(
    detection: ShuttleCourtDetection | None,
    corner_indices: list[int],
    args: argparse.Namespace,
    frame_shape: tuple[int, ...],
) -> tuple[np.ndarray | None, float, str]:
    if detection is None:
        return None, float("nan"), "no detection"
    if len(corner_indices) != 4:
        return None, float("nan"), "corner index count is not 4"
    if any(index < 0 or index >= len(detection.keypoints) for index in corner_indices):
        return None, float("nan"), "corner index out of range"

    raw_corners = detection.keypoints[corner_indices].astype(np.float32)
    raw_scores = np.ones((4,), dtype=np.float32)
    if detection.scores.size >= len(detection.keypoints):
        raw_scores = detection.scores[corner_indices].astype(np.float32)

    in_frame = valid_points(raw_corners, frame_shape)
    if not bool(np.all(in_frame)):
        return None, float(np.nanmean(raw_scores)), "invalid corner location"
    if not bool(np.all(raw_scores >= float(args.homography_kp_conf))):
        return None, float(np.nanmean(raw_scores)), "low corner confidence"

    try:
        ordered_corners, order = order_court_points(raw_corners)
    except ValueError as exc:
        return None, float(np.nanmean(raw_scores)), str(exc)

    if not is_reasonable_court_quad(ordered_corners, frame_shape, args.min_court_area_ratio):
        return None, float(np.nanmean(raw_scores[order])), "unreasonable court quadrilateral"
    return ordered_corners, float(np.nanmean(raw_scores[order])), "ok"


def fallback_or_invalid(
    state: HomographyState,
    reason: str,
    args: argparse.Namespace,
) -> CourtGeometryResult:
    max_fallback = max(0, int(args.max_fallback_frames))
    if state.last_valid_h is not None and state.fallback_frames < max_fallback:
        state.fallback_frames += 1
        return CourtGeometryResult(
            h=state.last_valid_h,
            raw_corners=None,
            smoothed_corners=state.last_valid_corners,
            valid=True,
            fallback=True,
            reason=reason,
            reproj_error=state.last_reproj_error,
        )
    if state.last_valid_h is not None:
        state.fallback_frames = max_fallback
        reason = f"{reason}; fallback expired"
    return CourtGeometryResult(
        h=None,
        raw_corners=None,
        smoothed_corners=None,
        valid=False,
        fallback=False,
        reason=reason,
    )


def update_court_geometry(
    detection: ShuttleCourtDetection | None,
    args: argparse.Namespace,
    corner_indices: list[int],
    frame_shape: tuple[int, ...],
    state: HomographyState,
) -> CourtGeometryResult:
    if not args.use_homography_overlay:
        return CourtGeometryResult(
            h=None,
            raw_corners=None,
            smoothed_corners=None,
            valid=False,
            fallback=False,
            reason="homography overlay disabled",
        )

    raw_corners, mean_corner_conf, reason = extract_ordered_corner_points(
        detection=detection,
        corner_indices=corner_indices,
        args=args,
        frame_shape=frame_shape,
    )
    if raw_corners is None:
        result = fallback_or_invalid(state, reason, args)
        result.mean_corner_conf = mean_corner_conf
        return result

    raw_h, raw_reproj_error, _ = estimate_homography(raw_corners, args.ransac_reproj_threshold)
    if raw_h is None or raw_reproj_error > float(args.max_reproj_error):
        result = fallback_or_invalid(state, f"bad homography reproj={raw_reproj_error:.1f}", args)
        result.raw_corners = raw_corners
        result.mean_corner_conf = mean_corner_conf
        result.reproj_error = raw_reproj_error
        return result

    height, width = frame_shape[:2]
    frame_diag = float(np.hypot(width, height))
    max_jump = max(20.0, frame_diag * max(0.0, float(args.max_corner_jump_ratio)))
    corner_jump = 0.0
    jump_is_large = False
    if state.last_valid_corners is not None:
        distances = np.linalg.norm(raw_corners - state.last_valid_corners, axis=1)
        corner_jump = float(np.mean(distances))
        jump_is_large = corner_jump > max_jump
        if jump_is_large and state.fallback_frames < max(0, int(args.max_fallback_frames)):
            result = fallback_or_invalid(state, f"corner jump {corner_jump:.1f}px", args)
            result.raw_corners = raw_corners
            result.mean_corner_conf = mean_corner_conf
            result.corner_jump = corner_jump
            return result

    alpha = float(np.clip(args.ema_alpha, 0.0, 1.0))
    if state.last_valid_corners is not None and not jump_is_large:
        smoothed_corners = alpha * raw_corners + (1.0 - alpha) * state.last_valid_corners
    else:
        smoothed_corners = raw_corners.copy()

    smoothed_h, smoothed_reproj_error, _ = estimate_homography(smoothed_corners, args.ransac_reproj_threshold)
    if smoothed_h is None or smoothed_reproj_error > float(args.max_reproj_error):
        result = fallback_or_invalid(state, f"bad smoothed homography reproj={smoothed_reproj_error:.1f}", args)
        result.raw_corners = raw_corners
        result.smoothed_corners = smoothed_corners
        result.mean_corner_conf = mean_corner_conf
        result.reproj_error = smoothed_reproj_error
        result.corner_jump = corner_jump
        return result

    state.last_valid_h = smoothed_h
    state.last_valid_corners = smoothed_corners
    state.last_reproj_error = raw_reproj_error
    state.fallback_frames = 0
    return CourtGeometryResult(
        h=smoothed_h,
        raw_corners=raw_corners,
        smoothed_corners=smoothed_corners,
        valid=True,
        fallback=False,
        reason="ok",
        reproj_error=raw_reproj_error,
        mean_corner_conf=mean_corner_conf,
        corner_jump=corner_jump,
    )


def draw_polyline(
    canvas: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    closed: bool,
) -> None:
    if len(points) < 2 or not np.isfinite(points).all():
        return
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=closed, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_template_overlay(
    canvas: np.ndarray,
    geometry: CourtGeometryResult,
    args: argparse.Namespace,
) -> None:
    if not geometry.valid or geometry.h is None:
        return
    line_color = (45, 245, 105) if not geometry.fallback else (100, 210, 130)
    shadow_color = (0, 70, 30)
    mask_color = (35, 210, 85) if not geometry.fallback else (70, 170, 95)
    thickness = max(1, int(args.line_thickness))

    if args.mask_alpha > 0:
        try:
            mask_polygon = project_points(STANDARD_COURT_TEMPLATE.mask_polygon, geometry.h)
        except cv2.error:
            mask_polygon = np.empty((0, 2), dtype=np.float32)
        if mask_polygon.shape[0] >= 3 and np.isfinite(mask_polygon).all():
            overlay = canvas.copy()
            polygon = np.asarray(mask_polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [polygon], mask_color)
            alpha = float(np.clip(args.mask_alpha, 0.0, 1.0))
            cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    try:
        projected_lines = project_court_template(geometry.h)
    except cv2.error:
        return

    for name, projected in projected_lines.items():
        if projected.shape[0] < 2 or not np.isfinite(projected).all():
            continue
        closed = name == "doubles_outer"
        line_thickness = thickness + 1 if closed else thickness
        draw_polyline(canvas, projected, shadow_color, line_thickness + 2, closed=closed)
        draw_polyline(canvas, projected, line_color, line_thickness, closed=closed)


def draw_index_polyline(
    canvas: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    indices: list[int],
    frame_shape: tuple[int, ...],
    color: tuple[int, int, int],
    thickness: int,
    kp_conf: float,
) -> None:
    if not indices:
        return
    visible = valid_points(points, frame_shape)
    selected: list[np.ndarray] = []
    for index in indices:
        if index >= len(points) or not visible[index]:
            continue
        if index < len(scores) and scores[index] < kp_conf:
            continue
        selected.append(points[index])
    if len(selected) >= 2:
        draw_polyline(canvas, np.asarray(selected, dtype=np.float32), color, thickness, closed=False)


def draw_raw_keypoints(
    canvas: np.ndarray,
    detection: ShuttleCourtDetection,
    args: argparse.Namespace,
    corner_indices: list[int],
    net_indices: list[int],
) -> None:
    points = detection.keypoints
    visible = valid_points(points, canvas.shape)
    corner_set = set(corner_indices)
    net_set = set(net_indices)
    corner_color = (45, 245, 105)
    net_color = (255, 180, 30)
    other_color = (210, 210, 210)
    low_color = (80, 80, 255)
    radius = max(3, int(args.point_radius) - 2)

    for index, point in enumerate(points):
        if not visible[index]:
            continue
        if index in net_set:
            color = net_color
        elif index in corner_set:
            color = corner_color
        else:
            color = other_color

        low_conf = index >= len(detection.scores) or detection.scores[index] < args.kp_conf
        fill_thickness = 2 if low_conf else -1
        if low_conf:
            color = low_color

        x, y = int(point[0]), int(point[1])
        cv2.circle(canvas, (x, y), radius + 2, (15, 15, 15), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, color, fill_thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius + 2, color, 1, lineType=cv2.LINE_AA)
        if args.show_labels:
            cv2.putText(canvas, str(index), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 2,
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def format_metric(value: float, digits: int = 2, suffix: str = "") -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}{suffix}"


def draw_detection(
    frame: np.ndarray,
    detection: ShuttleCourtDetection | None,
    geometry: CourtGeometryResult,
    args: argparse.Namespace,
    corner_indices: list[int],
    net_indices: list[int],
    fps_text: str,
) -> np.ndarray:
    canvas = frame.copy()
    net_color = (255, 180, 30)

    draw_template_overlay(canvas, geometry, args)
    if detection is not None:
        draw_index_polyline(
            canvas=canvas,
            points=detection.keypoints,
            scores=detection.scores,
            indices=net_indices,
            frame_shape=frame.shape,
            color=net_color,
            thickness=max(1, int(args.line_thickness)),
            kp_conf=args.kp_conf,
        )
        draw_raw_keypoints(canvas, detection, args, corner_indices, net_indices)

    box_conf = detection.box_conf if detection is not None else float("nan")
    kp_mean = float(np.nanmean(detection.scores)) if detection is not None and detection.scores.size else float("nan")
    h_state = "valid" if geometry.valid and not geometry.fallback else "fallback" if geometry.valid else "invalid"
    fallback_text = "yes" if geometry.fallback else "no"
    draw_text(
        canvas,
        (
            f"box conf {format_metric(box_conf)} | kp mean {format_metric(kp_mean)} | "
            f"corner mean {format_metric(geometry.mean_corner_conf)}"
        ),
        (24, 38),
        0.72,
    )
    draw_text(
        canvas,
        (
            f"H {h_state} | reproj {format_metric(geometry.reproj_error, 1, 'px')} | "
            f"fallback {fallback_text} | {geometry.reason}"
        ),
        (24, 72),
        0.64,
        color=(210, 245, 210) if geometry.valid else (80, 120, 255),
    )
    draw_text(canvas, fps_text, (24, 106), 0.64, color=(230, 230, 230))
    draw_text(
        canvas,
        f"template {args.template_mode} | corners TL,TR,BR,BL | q/Esc quit",
        (24, 140),
        0.58,
        color=(230, 230, 230),
    )
    return canvas


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Source video not found: {path}")
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return cap


def create_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter | None:
    if not path:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")
    return writer


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    corner_indices = parse_indices(args.corner_indices, expected=4)
    net_indices = parse_indices(args.net_indices, expected=4)
    homography_state = HomographyState()

    model = load_model(args.weights)
    cap = open_source(args.source)

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = create_writer(args.save_video, source_fps, (width, height)) if width and height else None

    print("=" * 72)
    print("ShuttleCourt realtime inference")
    print(f"source : {args.source}")
    print(f"weights: {args.weights}")
    print(f"device : {device}")
    print(f"points : corners {corner_indices} -> TL,TR,BR,BL | net {net_indices}")
    print(f"overlay: {args.template_mode} court template via homography")
    print("quit   : q or Esc")
    print("=" * 72)

    window_name = "ShuttleCourt realtime"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_id = 0
    started_at = time.perf_counter()
    last_frame_at = started_at
    max_frames = max(0, int(args.max_frames))
    log_every = max(1, int(args.log_every))

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if max_frames and frame_id >= max_frames:
                break

            loop_start = time.perf_counter()
            detection = infer_frame(model, frame, args, device)
            elapsed = time.perf_counter() - loop_start
            infer_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            avg_fps = (frame_id + 1) / max(time.perf_counter() - started_at, 1e-6)
            fps_text = f"infer {infer_fps:.1f} FPS | avg {avg_fps:.1f} FPS | frame {frame_id}"
            geometry = update_court_geometry(
                detection=detection,
                args=args,
                corner_indices=corner_indices,
                frame_shape=frame.shape,
                state=homography_state,
            )

            vis = draw_detection(
                frame=frame,
                detection=detection,
                geometry=geometry,
                args=args,
                corner_indices=corner_indices,
                net_indices=net_indices,
                fps_text=fps_text,
            )

            if writer is not None:
                writer.write(vis)

            if not args.no_display:
                cv2.imshow(window_name, vis)
                wait_ms = 1
                if args.realtime_playback:
                    frame_interval = 1.0 / source_fps
                    spent = time.perf_counter() - last_frame_at
                    wait_ms = max(1, int((frame_interval - spent) * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), 27):
                    break
                last_frame_at = time.perf_counter()

            if frame_id % log_every == 0:
                status = "detected" if detection is not None else "none"
                h_status = "fallback" if geometry.fallback else "valid" if geometry.valid else "invalid"
                reproj = format_metric(geometry.reproj_error, 1, "px")
                print(
                    f"[frame {frame_id:06d}] {status} | H {h_status} | "
                    f"reproj {reproj} | {geometry.reason} | {fps_text}"
                )

            frame_id += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyWindow(window_name)

    print(f"[done] processed {frame_id} frames")
    if args.save_video:
        print(f"[out] {args.save_video}")


if __name__ == "__main__":
    main()
