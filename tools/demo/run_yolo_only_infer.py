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


DEFAULT_SOURCE = (PROJECT_ROOT / "videos" / "MVI_0212.MP4").resolve()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_only_demo"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
WEIGHT_DIR_CANDIDATES = (
    PROJECT_ROOT / "weights" / "shttlecourtnet",
    PROJECT_ROOT / "weights" / "ShuttleCourtNet",
    PROJECT_ROOT / "assets" / "weights" / "ShuttleCourtNet",
)

# Direct-run configuration. Edit this block, then run:
#   python tools/demo/run_yolo_only_infer.py
#
# Command-line arguments are still supported and will override these defaults.
RUN_CONFIG: dict[str, Any] = {
    "source": r"F:\BDDataSet\Chen Long vs Son Wan Ho - MS Final [Denmark Open 2014].mp4",
    "weights": "",  # Empty means auto-detect ShuttleCourt weights.
    "output_dir": str(DEFAULT_OUTPUT_DIR),
    "device": "auto",
    "imgsz": 416,
    "conf": 0.25,
    "iou": 0.70,
    "max_det": 5,
    "stride": 1,
    "start_frame": 0,
    "max_frames": 0,
    "save_frames": False,
    "write_video": False,
    "display": True,
    "realtime_preview": True,
    "hide_summary": False,
    "log_every": 30,
    "retina_masks": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run raw YOLO inference only. This demo does not run court-line "
            "postprocessing, homography, white-line refinement, or template projection."
        )
    )
    parser.add_argument("--source", default=RUN_CONFIG["source"], help="Image, image folder, video path, or camera index.")
    parser.add_argument(
        "--weights",
        default=RUN_CONFIG["weights"],
        help="YOLO weight file or directory. Empty means auto-detect ShuttleCourt weights.",
    )
    parser.add_argument("--output-dir", default=RUN_CONFIG["output_dir"], help="Output directory.")
    parser.add_argument("--device", default=RUN_CONFIG["device"], help="auto, cpu, 0, cuda:0, ...")
    parser.add_argument("--imgsz", type=int, default=RUN_CONFIG["imgsz"], help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=RUN_CONFIG["conf"], help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=RUN_CONFIG["iou"], help="YOLO NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=RUN_CONFIG["max_det"], help="Maximum detections per frame.")
    parser.add_argument("--stride", type=int, default=RUN_CONFIG["stride"], help="For video sources, process every Nth frame.")
    parser.add_argument("--start-frame", type=int, default=RUN_CONFIG["start_frame"], help="For video sources, first source frame to process.")
    parser.add_argument("--max-frames", type=int, default=RUN_CONFIG["max_frames"], help="Maximum processed frames. 0 means no limit.")
    parser.add_argument(
        "--save-frames",
        action=argparse.BooleanOptionalAction,
        default=RUN_CONFIG["save_frames"],
        help="Save each annotated video frame as jpg.",
    )
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=RUN_CONFIG["write_video"],
        help="Write annotated mp4 for video sources.",
    )
    parser.add_argument("--no-video", dest="write_video", action="store_false", help="Do not write annotated mp4 for video sources.")
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=RUN_CONFIG["display"],
        help="Show an OpenCV preview window.",
    )
    parser.add_argument(
        "--realtime-preview",
        action=argparse.BooleanOptionalAction,
        default=RUN_CONFIG["realtime_preview"],
        help="When previewing a video file, pace display roughly by source FPS.",
    )
    parser.add_argument(
        "--hide-summary",
        action=argparse.BooleanOptionalAction,
        default=RUN_CONFIG["hide_summary"],
        help="Do not draw frame summary text over result.plot().",
    )
    parser.add_argument("--log-every", type=int, default=RUN_CONFIG["log_every"], help="Print progress every N processed video frames.")
    parser.add_argument(
        "--retina-masks",
        action=argparse.BooleanOptionalAction,
        dest="retina_masks",
        default=RUN_CONFIG["retina_masks"],
        help="Use high-resolution segmentation masks.",
    )
    return parser.parse_args()


def configure_ultralytics() -> None:
    config_dir = PROJECT_ROOT / ".ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_weights(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    candidates: list[Path] = []
    if str(raw).strip():
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(PROJECT_ROOT / path)
            candidates.append(path)
    else:
        candidates.extend(WEIGHT_DIR_CANDIDATES)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            exact = candidate / "ShuttleCourt.pt"
            if exact.is_file():
                return exact
            matches = sorted(candidate.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
            if matches:
                return matches[0]

    searched = [str(candidate) for candidate in candidates] or [str(path)]
    raise FileNotFoundError("Could not find YOLO weights. Searched: " + "; ".join(searched))


def load_model(weights: Path) -> Any:
    configure_ultralytics()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: install ultralytics first.") from exc
    return YOLO(str(weights))


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def source_stem(source: str) -> str:
    if source.isdigit():
        return f"camera_{source}"
    path = Path(source)
    return path.stem or "source"


def image_paths_from_source(source: str) -> list[Path]:
    if source.isdigit():
        return []
    path = Path(source)
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    return []


def predict_yolo(model: Any, frame: np.ndarray, args: argparse.Namespace, device: str) -> Any | None:
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


def render_result(frame: np.ndarray, result: Any | None, *, summary: str, hide_summary: bool) -> np.ndarray:
    if result is None:
        canvas = frame.copy()
    else:
        canvas = result.plot()
        if canvas is None:
            canvas = frame.copy()
    if not hide_summary:
        draw_text(canvas, summary, (24, 38), 0.70)
    return canvas


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (245, 245, 245),
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def result_record(result: Any | None, *, frame_id: int, timestamp_ms: int, infer_ms: float) -> dict[str, Any]:
    if result is None:
        return {
            "frame_id": int(frame_id),
            "timestamp_ms": int(timestamp_ms),
            "infer_ms": float(infer_ms),
            "detection_count": 0,
            "detections": [],
        }

    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    keypoints = getattr(result, "keypoints", None)
    names = getattr(result, "names", {}) or {}

    xyxy = to_numpy(getattr(boxes, "xyxy", None)) if boxes is not None else np.empty((0, 4), dtype=np.float32)
    conf = to_numpy(getattr(boxes, "conf", None)) if boxes is not None else np.empty((0,), dtype=np.float32)
    cls = to_numpy(getattr(boxes, "cls", None)) if boxes is not None else np.empty((0,), dtype=np.float32)
    mask_xy = list(getattr(masks, "xy", []) or []) if masks is not None else []
    kpt_xy = to_numpy(getattr(keypoints, "xy", None)) if keypoints is not None else np.empty((0, 0, 2), dtype=np.float32)
    kpt_conf = to_numpy(getattr(keypoints, "conf", None)) if keypoints is not None else np.empty((0, 0), dtype=np.float32)

    count = max(len(xyxy), len(mask_xy), int(kpt_xy.shape[0]) if kpt_xy.ndim == 3 else 0)
    detections = []
    for index in range(count):
        class_id = int(cls[index]) if index < len(cls) else None
        detections.append(
            {
                "index": int(index),
                "class_id": class_id,
                "name": str(names.get(class_id, class_id)) if class_id is not None else "",
                "confidence": float(conf[index]) if index < len(conf) else None,
                "bbox_xyxy": _float_list(xyxy[index]) if index < len(xyxy) else [],
                "mask_polygon_xy": _points_list(mask_xy[index]) if index < len(mask_xy) else [],
                "keypoints": _keypoints_list(kpt_xy, kpt_conf, index),
            }
        )

    return {
        "frame_id": int(frame_id),
        "timestamp_ms": int(timestamp_ms),
        "infer_ms": float(infer_ms),
        "detection_count": int(count),
        "detections": detections,
    }


def _float_list(values: Any) -> list[float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return [round(float(value), 4) for value in array.tolist()]


def _points_list(values: Any) -> list[list[float]]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return []
    array = array.reshape(-1, 2)
    return [[round(float(x), 4), round(float(y), 4)] for x, y in array.tolist()]


def _keypoints_list(kpt_xy: np.ndarray, kpt_conf: np.ndarray, index: int) -> list[dict[str, float]]:
    if kpt_xy.ndim != 3 or index >= kpt_xy.shape[0]:
        return []
    points = np.asarray(kpt_xy[index], dtype=np.float32).reshape(-1, 2)
    scores = (
        np.asarray(kpt_conf[index], dtype=np.float32).reshape(-1)
        if kpt_conf.ndim == 2 and index < kpt_conf.shape[0]
        else np.ones((points.shape[0],), dtype=np.float32)
    )
    return [
        {
            "index": int(point_index),
            "x": round(float(point[0]), 4),
            "y": round(float(point[1]), 4),
            "confidence": round(float(scores[point_index]), 4) if point_index < len(scores) else 1.0,
        }
        for point_index, point in enumerate(points)
    ]


def process_images(
    model: Any,
    image_paths: list[Path],
    args: argparse.Namespace,
    device: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, path in enumerate(image_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[warn] failed to read image: {path}")
            continue

        started_at = time.perf_counter()
        result = predict_yolo(model, frame, args, device)
        infer_ms = (time.perf_counter() - started_at) * 1000.0
        record = result_record(result, frame_id=index, timestamp_ms=0, infer_ms=infer_ms)
        out_path = output_dir / f"{path.stem}_yolo_only.jpg"
        summary = f"{path.name} | detections {record['detection_count']} | infer {infer_ms:.1f} ms"
        vis = render_result(frame, result, summary=summary, hide_summary=bool(args.hide_summary))
        cv2.imwrite(str(out_path), vis)
        record["source"] = str(path)
        record["visualization"] = str(out_path)
        records.append(record)
        print(f"[image] {path.name} detections={record['detection_count']} infer={infer_ms:.1f}ms -> {out_path}")
    return records


def process_video(
    model: Any,
    args: argparse.Namespace,
    device: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    source = str(args.source)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stride = max(1, int(args.stride))
    start_frame = max(0, int(args.start_frame))
    max_frames = max(0, int(args.max_frames))
    stem = source_stem(source)

    if start_frame > 0 and not source.isdigit():
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        source_frame_id = start_frame
    else:
        source_frame_id = 0

    writer = None
    if args.write_video and width > 0 and height > 0:
        video_path = output_dir / f"{stem}_yolo_only.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, fps / float(stride)),
            (width, height),
        )
        if writer.isOpened():
            print(f"[out] video: {video_path}")
        else:
            writer.release()
            writer = None
            print(f"[warn] failed to create video writer: {video_path}")

    if args.display:
        cv2.namedWindow("YOLO only inference", cv2.WINDOW_NORMAL)

    records: list[dict[str, Any]] = []
    processed = 0
    detected_frames = 0
    frame_interval_ms = 1000.0 / max(1.0, fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if (source_frame_id - start_frame) % stride != 0:
                source_frame_id += 1
                continue
            if max_frames and processed >= max_frames:
                break

            timestamp_ms = int(round(cap.get(cv2.CAP_PROP_POS_MSEC) or (source_frame_id * 1000.0 / fps)))
            loop_started_at = time.perf_counter()
            started_at = loop_started_at
            result = predict_yolo(model, frame, args, device)
            infer_ms = (time.perf_counter() - started_at) * 1000.0
            record = result_record(result, frame_id=source_frame_id, timestamp_ms=timestamp_ms, infer_ms=infer_ms)
            detected_frames += int(record["detection_count"] > 0)

            summary = (
                f"frame {source_frame_id} | t {timestamp_ms} ms | "
                f"detections {record['detection_count']} | infer {infer_ms:.1f} ms"
            )
            vis = render_result(frame, result, summary=summary, hide_summary=bool(args.hide_summary))
            if writer is not None:
                writer.write(vis)
            if args.save_frames:
                frame_path = output_dir / "frames" / f"{stem}_{source_frame_id:06d}_yolo_only.jpg"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frame_path), vis)
                record["visualization"] = str(frame_path)

            if args.display:
                cv2.imshow("YOLO only inference", vis)
                spent_ms = (time.perf_counter() - loop_started_at) * 1000.0
                wait_ms = 1
                if args.realtime_preview and not source.isdigit():
                    wait_ms = max(1, int(round(frame_interval_ms - spent_ms)))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), 27):
                    break

            records.append(record)
            if processed % max(1, int(args.log_every)) == 0:
                print(
                    f"[frame {source_frame_id:06d}] detections={record['detection_count']} "
                    f"infer={infer_ms:.1f}ms"
                )

            processed += 1
            source_frame_id += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyWindow("YOLO only inference")

    print(f"[done] processed={processed} detected_frames={detected_frames}")
    return records


def write_summary(output_dir: Path, source: str, weights: Path, device: str, args: argparse.Namespace, records: list[dict[str, Any]]) -> Path:
    payload = {
        "source": source,
        "weights": str(weights),
        "device": device,
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
        "save_frames": bool(args.save_frames),
        "write_video": bool(args.write_video),
        "display": bool(args.display),
        "realtime_preview": bool(args.realtime_preview),
        "retina_masks": bool(args.retina_masks),
        "note": "Raw YOLO inference only; no court-line postprocessing or homography is applied.",
        "records": records,
    }
    summary_path = output_dir / f"{source_stem(source)}_yolo_only.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = str(args.source)
    weights = resolve_weights(args.weights)
    device = resolve_device(str(args.device))
    model = load_model(weights)
    task = getattr(model, "task", "")
    names = getattr(model, "names", {})

    print("=" * 72)
    print("YOLO only inference demo")
    print(f"source : {source}")
    print(f"weights: {weights}")
    print(f"task   : {task}")
    print(f"names  : {names}")
    print(f"device : {device}")
    print(f"output : {output_dir}")
    print("post   : disabled court-line fitting / homography / template projection")
    print("=" * 72)

    image_paths = image_paths_from_source(source)
    if image_paths:
        records = process_images(model, image_paths, args, device, output_dir)
    else:
        records = process_video(model, args, device, output_dir)

    summary_path = write_summary(output_dir, source, weights, device, args, records)
    print(f"[out] json: {summary_path}")


if __name__ == "__main__":
    main()
