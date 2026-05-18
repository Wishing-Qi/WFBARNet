from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SOURCE = PROJECT_ROOT / "videos" / "MVI_0212.MP4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shuttlecourt_seg_video_demo"
WEIGHT_DIR_CANDIDATES = (
    PROJECT_ROOT / "weights" / "shttlecourtnet",
    PROJECT_ROOT / "weights" / "ShuttleCourtNet",
    PROJECT_ROOT / "assets" / "weights" / "ShuttleCourtNet",
)
MASK_COLOR = (45, 245, 105)
BBOX_COLOR = (255, 180, 30)
TEXT_COLOR = (245, 245, 245)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a video with the ShuttleCourtNet YOLO segmentation model and save annotated frames."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Input video path.")
    parser.add_argument(
        "--weights",
        default="",
        help="Weight file or directory. Auto-detects weights/shttlecourtnet or assets/weights/ShuttleCourtNet.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for output frames and JSON.")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, cuda:0, ...")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO object confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=3, help="Maximum court detections per sampled frame.")
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame to sample.")
    parser.add_argument("--stride", type=int, default=120, help="Sample every N source frames.")
    parser.add_argument("--frames", type=int, default=8, help="Number of annotated frames to save.")
    parser.add_argument("--mask-alpha", type=float, default=0.28, help="Segmentation mask opacity.")
    parser.add_argument("--line-thickness", type=int, default=3, help="Mask outline thickness.")
    parser.add_argument("--hide-boxes", action="store_true", help="Do not draw boxes around detected courts.")
    parser.add_argument("--hide-labels", action="store_true", help="Do not draw labels or frame status text.")
    parser.add_argument(
        "--no-retina-masks",
        dest="retina_masks",
        action="store_false",
        help="Disable high-resolution mask coordinates.",
    )
    parser.set_defaults(retina_masks=True)
    return parser.parse_args()


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


def resolve_weights(raw: str) -> Path:
    if raw:
        path = Path(raw)
        if path.is_file():
            return path
        if path.is_dir():
            matches = sorted(path.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
            if matches:
                return matches[0]
        raise FileNotFoundError(f"Weight path not found or contains no .pt files: {path}")

    for directory in WEIGHT_DIR_CANDIDATES:
        exact = directory / "ShuttleCourt.pt"
        if exact.is_file():
            return exact
        if directory.is_dir():
            matches = sorted(directory.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
            if matches:
                return matches[0]

    searched = "\n".join(f"- {path}" for path in WEIGHT_DIR_CANDIDATES)
    raise FileNotFoundError(f"Could not find ShuttleCourtNet weights. Searched:\n{searched}")


def load_model(weights: Path) -> Any:
    configure_ultralytics()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install ultralytics") from exc
    return YOLO(str(weights))


def open_video(source: str) -> cv2.VideoCapture:
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Source video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {path}")
    return cap


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


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.64,
    color: tuple[int, int, int] = TEXT_COLOR,
    thickness: int = 2,
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def result_objects(result: Any) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    names = getattr(result, "names", {}) or {}
    xyxy = to_numpy(getattr(boxes, "xyxy", None)) if boxes is not None else np.empty((0, 4), dtype=np.float32)
    cls = to_numpy(getattr(boxes, "cls", None)) if boxes is not None else np.empty((0,), dtype=np.float32)
    conf = to_numpy(getattr(boxes, "conf", None)) if boxes is not None else np.empty((0,), dtype=np.float32)
    mask_xy = list(getattr(masks, "xy", []) or []) if masks is not None else []

    count = max(len(xyxy), len(mask_xy))
    objects: list[dict[str, Any]] = []
    for index in range(count):
        cls_id = int(cls[index]) if index < len(cls) else None
        confidence = float(conf[index]) if index < len(conf) else None
        polygon = np.asarray(mask_xy[index], dtype=np.float32) if index < len(mask_xy) else np.empty((0, 2), dtype=np.float32)
        bbox = np.asarray(xyxy[index], dtype=np.float32) if index < len(xyxy) else bbox_from_polygon(polygon)
        area = float(cv2.contourArea(polygon.reshape(-1, 1, 2))) if len(polygon) >= 3 else 0.0
        objects.append(
            {
                "index": int(index),
                "class_id": cls_id,
                "name": names.get(cls_id, str(cls_id)) if cls_id is not None else None,
                "confidence": confidence,
                "bbox_xyxy": [round(float(value), 3) for value in bbox.tolist()] if bbox.size == 4 else [],
                "mask_area": round(area, 3),
                "polygon_xy": [[round(float(x), 3), round(float(y), 3)] for x, y in polygon.tolist()],
            }
        )
    return objects


def bbox_from_polygon(polygon: np.ndarray) -> np.ndarray:
    if polygon.size == 0:
        return np.empty((0,), dtype=np.float32)
    x1, y1 = np.min(polygon, axis=0)
    x2, y2 = np.max(polygon, axis=0)
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def draw_segmentation(
    frame: np.ndarray,
    objects: list[dict[str, Any]],
    args: argparse.Namespace,
    frame_id: int,
    timestamp_ms: int,
    infer_ms: float,
) -> np.ndarray:
    canvas = frame.copy()
    overlay = canvas.copy()
    alpha = float(np.clip(args.mask_alpha, 0.0, 1.0))
    thickness = max(1, int(args.line_thickness))

    for obj in objects:
        polygon = np.asarray(obj["polygon_xy"], dtype=np.float32)
        if len(polygon) >= 3:
            pts = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], MASK_COLOR)
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 70, 30), thickness=thickness + 3, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [pts], isClosed=True, color=MASK_COLOR, thickness=thickness, lineType=cv2.LINE_AA)

    if alpha > 0.0 and any(len(obj["polygon_xy"]) >= 3 for obj in objects):
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        for obj in objects:
            polygon = np.asarray(obj["polygon_xy"], dtype=np.float32)
            if len(polygon) >= 3:
                pts = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], isClosed=True, color=(0, 70, 30), thickness=thickness + 3, lineType=cv2.LINE_AA)
                cv2.polylines(canvas, [pts], isClosed=True, color=MASK_COLOR, thickness=thickness, lineType=cv2.LINE_AA)

    for obj in objects:
        bbox = obj.get("bbox_xyxy") or []
        if len(bbox) == 4 and not args.hide_boxes:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (10, 35, 10), thickness + 2, lineType=cv2.LINE_AA)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), BBOX_COLOR, thickness, lineType=cv2.LINE_AA)
        if not args.hide_labels:
            label = obj.get("name") or "court"
            if obj.get("confidence") is not None:
                label = f"{label} {float(obj['confidence']):.2f}"
            origin = (24, 112 + int(obj["index"]) * 30)
            draw_text(canvas, label, origin, 0.58, color=BBOX_COLOR, thickness=2)

    if not args.hide_labels:
        draw_text(canvas, f"frame {frame_id} | t {timestamp_ms} ms | detections {len(objects)}", (24, 38), 0.72)
        draw_text(canvas, f"imgsz {args.imgsz} | conf {args.conf:.2f} | infer {infer_ms:.1f} ms", (24, 72), 0.64)
    return canvas


def predict_frame(model: Any, frame: np.ndarray, args: argparse.Namespace, device: str) -> Any | None:
    results = model.predict(
        frame,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=max(1, int(args.max_det)),
        device=device,
        retina_masks=bool(args.retina_masks),
        verbose=False,
    )
    return results[0] if results else None


def main() -> None:
    args = parse_args()
    weights = resolve_weights(args.weights)
    output_dir = Path(args.output_dir)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(args.device))
    model = load_model(weights)
    task = getattr(model, "task", None)
    names = getattr(model, "names", {})
    if task != "segment":
        print(f"[warn] loaded YOLO task is {task!r}, expected 'segment'. The demo will still try to draw masks.")

    cap = open_video(str(args.source))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(args.start_frame))
    stride = max(1, int(args.stride))
    target_frames = max(1, int(args.frames))
    source_stem = Path(str(args.source)).stem

    print("=" * 72)
    print("ShuttleCourtNet segmentation video frame demo")
    print(f"source : {args.source}")
    print(f"weights: {weights}")
    print(f"task   : {task}")
    print(f"names  : {names}")
    print(f"output : {frame_dir}")
    print(f"device : {device}")
    print(f"sample : start={start_frame}, stride={stride}, frames={target_frames}")
    print("=" * 72)

    records: list[dict[str, Any]] = []
    saved = 0
    try:
        for sample_index in range(target_frames):
            frame_id = start_frame + sample_index * stride
            if total_frames and frame_id >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[warn] failed to read frame {frame_id}")
                continue

            actual_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or (frame_id + 1)) - 1
            timestamp_ms = int(round(cap.get(cv2.CAP_PROP_POS_MSEC) or (actual_frame_id * 1000.0 / source_fps)))

            started_at = time.perf_counter()
            result = predict_frame(model, frame, args, device)
            infer_ms = (time.perf_counter() - started_at) * 1000.0
            objects = result_objects(result) if result is not None else []
            vis = draw_segmentation(frame, objects, args, actual_frame_id, timestamp_ms, infer_ms)

            out_path = frame_dir / f"{source_stem}_{actual_frame_id:06d}_shuttlecourt_seg.jpg"
            cv2.imwrite(str(out_path), vis)
            records.append(
                {
                    "frame_id": int(actual_frame_id),
                    "timestamp_ms": int(timestamp_ms),
                    "infer_ms": float(infer_ms),
                    "image": str(out_path),
                    "object_count": len(objects),
                    "objects": objects,
                }
            )
            saved += 1

            best_conf = max((obj["confidence"] for obj in objects if obj["confidence"] is not None), default=0.0)
            print(
                f"[{saved:02d}/{target_frames}] frame={actual_frame_id:06d} "
                f"detections={len(objects)} best={best_conf:.3f} -> {out_path}"
            )
    finally:
        cap.release()

    payload = {
        "source": str(args.source),
        "weights": str(weights),
        "task": task,
        "names": names,
        "device": device,
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "start_frame": start_frame,
        "stride": stride,
        "requested_frames": target_frames,
        "saved_frames": saved,
        "records": records,
    }
    summary_path = output_dir / f"{source_stem}_shuttlecourt_seg_frames.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] frames : {frame_dir}")
    print(f"[out] summary: {summary_path}")


if __name__ == "__main__":
    main()
