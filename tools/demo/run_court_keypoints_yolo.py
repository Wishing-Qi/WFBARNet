from __future__ import annotations

import argparse
import json
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

from tools.demo.run_shuttlecourt_realtime_gpu import (  # noqa: E402
    DEFAULT_WEIGHTS,
    draw_text,
    load_model,
    resolve_device,
    to_numpy,
    valid_points,
)


DEFAULT_SOURCE = PROJECT_ROOT / "videos" / "MVI_0212.MP4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shuttlecourt_demo"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
PALETTE = (
    (45, 245, 105),
    (255, 180, 30),
    (80, 190, 255),
    (230, 120, 255),
    (255, 235, 80),
)


@dataclass(slots=True)
class CourtKeypointCandidate:
    index: int
    bbox: np.ndarray
    box_conf: float
    keypoints: np.ndarray
    scores: np.ndarray
    rank_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline raw YOLO keypoint demo for ShuttleCourt.pt."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image, image folder, video path, or camera index.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="ShuttleCourt YOLO pose weights.")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, cuda:0, ...")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size. The bundled model was trained at 1280.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO object confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=5, help="Maximum detections per frame.")
    parser.add_argument("--kp-conf", type=float, default=0.15, help="Keypoints below this score are drawn hollow/red.")
    parser.add_argument("--stride", type=int, default=30, help="For videos, process every Nth source frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum processed frames. 0 means no limit.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for visualization and JSON outputs.")
    parser.add_argument("--no-video", action="store_true", help="Do not write annotated mp4 for video sources.")
    parser.add_argument("--save-frames", action="store_true", help="Save each processed video frame as a jpg.")
    parser.add_argument("--display", action="store_true", help="Show an OpenCV window while processing.")
    parser.add_argument("--hide-labels", action="store_true", help="Hide raw keypoint index labels.")
    parser.add_argument("--draw-all", action="store_true", help="Draw every detection instead of only the highest-ranked one.")
    parser.add_argument(
        "--court-indices",
        "--corner-indices",
        dest="court_indices",
        default="0,1,2,3",
        help="Keypoint indices to connect as the court/corner group.",
    )
    parser.add_argument("--net-indices", default="4,5,6,7", help="Keypoint indices to connect as the net group.")
    parser.add_argument("--surface-indices", default="0,2,3,1", help="Four indices used for the translucent raw surface mask.")
    parser.add_argument("--no-mask", action="store_true", help="Disable translucent raw surface mask.")
    parser.add_argument("--mask-alpha", type=float, default=0.18, help="Raw surface mask opacity.")
    parser.add_argument("--line-thickness", type=int, default=3)
    parser.add_argument("--point-radius", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N processed frames.")
    return parser.parse_args()


def parse_indices(raw: str) -> list[int]:
    indices: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            indices.append(int(part))
    return indices


def source_stem(source: str) -> str:
    if source.isdigit():
        return f"camera_{source}"
    path = Path(source)
    return path.stem or "source"


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def list_image_paths(path: Path) -> list[Path]:
    if path.is_file() and is_image_path(path):
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.is_file() and is_image_path(item))
    return []


def predict_candidates(
    model: Any,
    frame: np.ndarray,
    args: argparse.Namespace,
    device: str,
) -> list[CourtKeypointCandidate]:
    results = model.predict(
        frame,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=max(1, int(args.max_det)),
        device=device,
        verbose=False,
    )
    if not results:
        return []

    result = results[0]
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return []

    xy = to_numpy(getattr(keypoints, "xy", None))
    if xy.size == 0 or xy.ndim != 3:
        return []

    kp_conf = to_numpy(getattr(keypoints, "conf", None))
    if kp_conf.size == 0:
        kp_conf = np.ones(xy.shape[:2], dtype=np.float32)

    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        box_xyxy = to_numpy(getattr(boxes, "xyxy", None))
        box_conf = to_numpy(getattr(boxes, "conf", None))
    else:
        box_xyxy = np.empty((0, 4), dtype=np.float32)
        box_conf = np.empty((0,), dtype=np.float32)

    count = int(xy.shape[0])
    candidates: list[CourtKeypointCandidate] = []
    for index in range(count):
        bbox = (
            np.asarray(box_xyxy[index], dtype=np.float32)
            if index < len(box_xyxy)
            else bbox_from_keypoints(xy[index], frame.shape)
        )
        conf = float(box_conf[index]) if index < len(box_conf) else 1.0
        scores = np.asarray(kp_conf[index], dtype=np.float32)
        finite_scores = scores[np.isfinite(scores)]
        kp_mean = float(np.mean(finite_scores)) if finite_scores.size else 0.0
        rank_score = conf * kp_mean
        candidates.append(
            CourtKeypointCandidate(
                index=index,
                bbox=bbox.reshape(4).astype(np.float32),
                box_conf=conf,
                keypoints=np.asarray(xy[index], dtype=np.float32),
                scores=scores,
                rank_score=rank_score,
            )
        )
    candidates.sort(key=lambda item: item.rank_score, reverse=True)
    return candidates


def bbox_from_keypoints(points: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    height, width = frame_shape[:2]
    visible = valid_points(points, frame_shape)
    if not bool(np.any(visible)):
        return np.array([0.0, 0.0, float(width - 1), float(height - 1)], dtype=np.float32)
    selected = np.asarray(points[visible], dtype=np.float32)
    x1, y1 = np.min(selected, axis=0)
    x2, y2 = np.max(selected, axis=0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def draw_candidates(
    frame: np.ndarray,
    candidates: list[CourtKeypointCandidate],
    args: argparse.Namespace,
    court_indices: list[int],
    net_indices: list[int],
    surface_indices: list[int],
    frame_id: int,
    timestamp_ms: int,
) -> np.ndarray:
    canvas = frame.copy()
    draw_items = candidates if args.draw_all else candidates[:1]

    for draw_index, candidate in enumerate(draw_items):
        color = PALETTE[draw_index % len(PALETTE)]
        draw_raw_surface(canvas, candidate, surface_indices, args, color)
        draw_index_polyline(canvas, candidate, court_indices, args, color, closed=True)
        draw_index_polyline(canvas, candidate, net_indices, args, (255, 180, 30), closed=False)
        draw_candidate_box(canvas, candidate, color, draw_index)
        draw_raw_keypoints(canvas, candidate, args, court_indices, net_indices, color)

    best = candidates[0] if candidates else None
    summary = (
        f"frame {frame_id} | t {timestamp_ms}ms | detections {len(candidates)}"
        if best is None
        else (
            f"frame {frame_id} | t {timestamp_ms}ms | detections {len(candidates)} | "
            f"best box {best.box_conf:.2f} kp {candidate_kp_mean(best):.2f} score {best.rank_score:.2f}"
        )
    )
    draw_text(canvas, summary, (24, 38), 0.72)
    draw_text(
        canvas,
        f"imgsz {args.imgsz} | conf {args.conf:.2f} | kp-conf {args.kp_conf:.2f} | labels {'off' if args.hide_labels else 'on'}",
        (24, 72),
        0.60,
        color=(225, 235, 245),
    )
    return canvas


def draw_raw_surface(
    canvas: np.ndarray,
    candidate: CourtKeypointCandidate,
    indices: list[int],
    args: argparse.Namespace,
    color: tuple[int, int, int],
) -> None:
    if args.no_mask or len(indices) < 3:
        return
    points = points_for_indices(candidate, indices, canvas.shape, args.kp_conf)
    if len(points) < 3:
        return
    polygon = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [polygon], color)
    alpha = float(np.clip(args.mask_alpha, 0.0, 1.0))
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)


def draw_index_polyline(
    canvas: np.ndarray,
    candidate: CourtKeypointCandidate,
    indices: list[int],
    args: argparse.Namespace,
    color: tuple[int, int, int],
    *,
    closed: bool,
) -> None:
    points = points_for_indices(candidate, indices, canvas.shape, args.kp_conf)
    if len(points) < 2:
        return
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    thickness = max(1, int(args.line_thickness))
    cv2.polylines(canvas, [pts], isClosed=closed, color=(10, 35, 10), thickness=thickness + 3, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [pts], isClosed=closed, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def points_for_indices(
    candidate: CourtKeypointCandidate,
    indices: list[int],
    frame_shape: tuple[int, ...],
    kp_conf: float,
) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    visible = valid_points(candidate.keypoints, frame_shape)
    for index in indices:
        if index < 0 or index >= len(candidate.keypoints):
            continue
        if not bool(visible[index]):
            continue
        if index < len(candidate.scores) and float(candidate.scores[index]) < float(kp_conf):
            continue
        points.append(candidate.keypoints[index])
    return points


def draw_candidate_box(
    canvas: np.ndarray,
    candidate: CourtKeypointCandidate,
    color: tuple[int, int, int],
    draw_index: int,
) -> None:
    x1, y1, x2, y2 = [int(round(float(value))) for value in candidate.bbox[:4]]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (8, 28, 8), 4, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
    label = f"#{draw_index} box {candidate.box_conf:.2f} kp {candidate_kp_mean(candidate):.2f}"
    draw_text(canvas, label, (max(8, x1), max(24, y1 - 8)), 0.55, color=color)


def draw_raw_keypoints(
    canvas: np.ndarray,
    candidate: CourtKeypointCandidate,
    args: argparse.Namespace,
    court_indices: list[int],
    net_indices: list[int],
    default_color: tuple[int, int, int],
) -> None:
    visible = valid_points(candidate.keypoints, canvas.shape)
    court_set = set(court_indices)
    net_set = set(net_indices)
    radius = max(2, int(args.point_radius))
    for index, point in enumerate(candidate.keypoints):
        if not bool(visible[index]):
            continue
        score = float(candidate.scores[index]) if index < len(candidate.scores) else 1.0
        if index in net_set:
            color = (255, 180, 30)
        elif index in court_set:
            color = default_color
        else:
            color = (220, 220, 220)
        low_conf = score < float(args.kp_conf)
        if low_conf:
            color = (80, 80, 255)
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        cv2.circle(canvas, (x, y), radius + 2, (10, 10, 10), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, color, 2 if low_conf else -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius + 2, color, 1, lineType=cv2.LINE_AA)
        if not args.hide_labels:
            text = f"{index}:{score:.2f}"
            draw_text(canvas, text, (x + radius + 4, y - radius - 2), 0.45, color=color, thickness=1)


def candidate_kp_mean(candidate: CourtKeypointCandidate) -> float:
    scores = candidate.scores[np.isfinite(candidate.scores)]
    return float(np.mean(scores)) if scores.size else 0.0


def candidate_record(candidate: CourtKeypointCandidate) -> dict[str, Any]:
    return {
        "index": int(candidate.index),
        "bbox": [float(value) for value in candidate.bbox.tolist()],
        "box_conf": float(candidate.box_conf),
        "keypoint_mean_conf": candidate_kp_mean(candidate),
        "rank_score": float(candidate.rank_score),
        "keypoints": [
            {
                "index": int(index),
                "x": float(point[0]),
                "y": float(point[1]),
                "conf": float(candidate.scores[index]) if index < len(candidate.scores) else 1.0,
            }
            for index, point in enumerate(candidate.keypoints)
        ],
    }


def frame_record(
    frame_id: int,
    timestamp_ms: int,
    candidates: list[CourtKeypointCandidate],
) -> dict[str, Any]:
    return {
        "frame_id": int(frame_id),
        "timestamp_ms": int(timestamp_ms),
        "detection_count": len(candidates),
        "best_score": float(candidates[0].rank_score) if candidates else 0.0,
        "detections": [candidate_record(candidate) for candidate in candidates],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_image_paths(
    model: Any,
    image_paths: list[Path],
    args: argparse.Namespace,
    device: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    court_indices = parse_indices(args.court_indices)
    net_indices = parse_indices(args.net_indices)
    surface_indices = parse_indices(args.surface_indices)
    records: list[dict[str, Any]] = []

    for frame_id, path in enumerate(image_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[warn] failed to read image: {path}")
            continue
        started_at = time.perf_counter()
        candidates = predict_candidates(model, frame, args, device)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        vis = draw_candidates(frame, candidates, args, court_indices, net_indices, surface_indices, frame_id, 0)
        out_path = output_dir / f"{path.stem}_shuttlecourt.jpg"
        cv2.imwrite(str(out_path), vis)
        record = frame_record(frame_id, 0, candidates)
        record["source"] = str(path)
        record["infer_ms"] = elapsed_ms
        record["visualization"] = str(out_path)
        records.append(record)
        print(f"[image] {path.name} detections={len(candidates)} infer={elapsed_ms:.1f}ms -> {out_path}")
    return records


def process_video_source(
    model: Any,
    args: argparse.Namespace,
    device: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    source = str(args.source)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    court_indices = parse_indices(args.court_indices)
    net_indices = parse_indices(args.net_indices)
    surface_indices = parse_indices(args.surface_indices)
    stem = source_stem(source)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stride = max(1, int(args.stride))
    max_frames = max(0, int(args.max_frames))
    log_every = max(1, int(args.log_every))
    writer = None
    if not args.no_video and width > 0 and height > 0:
        video_path = output_dir / f"{stem}_shuttlecourt.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, fps / float(stride)),
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            writer = None
            print(f"[warn] could not create video writer: {video_path}")
        else:
            print(f"[out] video: {video_path}")

    if args.display:
        cv2.namedWindow("ShuttleCourt YOLO raw check", cv2.WINDOW_NORMAL)

    records: list[dict[str, Any]] = []
    source_frame_id = 0
    processed = 0
    detected = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if source_frame_id % stride != 0:
                source_frame_id += 1
                continue
            if max_frames and processed >= max_frames:
                break

            timestamp_ms = int(round(cap.get(cv2.CAP_PROP_POS_MSEC) or (source_frame_id * 1000.0 / fps)))
            started_at = time.perf_counter()
            candidates = predict_candidates(model, frame, args, device)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            detected += int(bool(candidates))
            vis = draw_candidates(
                frame,
                candidates,
                args,
                court_indices,
                net_indices,
                surface_indices,
                source_frame_id,
                timestamp_ms,
            )
            if writer is not None:
                writer.write(vis)
            if args.save_frames:
                frame_path = output_dir / "frames" / f"{stem}_{source_frame_id:06d}.jpg"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frame_path), vis)
            if args.display:
                cv2.imshow("ShuttleCourt YOLO raw check", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            record = frame_record(source_frame_id, timestamp_ms, candidates)
            record["infer_ms"] = elapsed_ms
            records.append(record)

            if processed % log_every == 0:
                best = candidates[0].rank_score if candidates else 0.0
                print(
                    f"[frame {source_frame_id:06d}] detections={len(candidates)} "
                    f"best={best:.3f} infer={elapsed_ms:.1f}ms"
                )

            processed += 1
            source_frame_id += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyWindow("ShuttleCourt YOLO raw check")

    print(f"[done] processed={processed} detected={detected} source={source}")
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))
    model = load_model(str(args.weights))

    source = str(args.source)
    image_paths = [] if source.isdigit() else list_image_paths(Path(source))
    if image_paths:
        records = process_image_paths(model, image_paths, args, device, output_dir)
    else:
        records = process_video_source(model, args, device, output_dir)

    payload = {
        "source": source,
        "weights": str(args.weights),
        "device": device,
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "kp_conf": float(args.kp_conf),
        "court_indices": parse_indices(args.court_indices),
        "net_indices": parse_indices(args.net_indices),
        "surface_indices": parse_indices(args.surface_indices),
        "records": records,
    }
    json_path = output_dir / f"{source_stem(source)}_shuttlecourt.json"
    write_json(json_path, payload)
    print(f"[out] json: {json_path}")


if __name__ == "__main__":
    main()
