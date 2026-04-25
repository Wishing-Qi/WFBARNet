from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Iterator

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tracknet_v3 import TrackNetV3


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TrackNetV3 PyTorch checkpoint to a TensorRT INT8 engine."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "assets" / "weights" / "track" / "model_best.pt"),
        help="TrackNetV3 PyTorch checkpoint path.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "assets" / "weights" / "track" / "tracknetv3_int8.engine"),
        help="Output TensorRT engine path.",
    )
    parser.add_argument(
        "--onnx",
        default="",
        help="Intermediate ONNX path. Defaults to output path with .onnx suffix.",
    )
    parser.add_argument(
        "--calib-source",
        default=str(PROJECT_ROOT / "videos"),
        help="Calibration video, image folder, or folder containing videos/images.",
    )
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--calib-batch-size", type=int, default=8)
    parser.add_argument("--calib-batches", type=int, default=128)
    parser.add_argument(
        "--calib-stride",
        type=int,
        default=3,
        help="Use every Nth frame window for calibration.",
    )
    parser.add_argument(
        "--calib-seconds",
        type=float,
        default=120.0,
        help="Use at most this many seconds from each calibration video.",
    )
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="Also allow FP16 tactics while building INT8.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_tracknet(checkpoint_path: Path, device: str) -> TrackNetV3:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = TrackNetV3().to(device).eval()
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    cleaned = {str(k).replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"[warn] checkpoint load: missing={len(missing)} unexpected={len(unexpected)}")
    return model


def export_onnx(
    model: TrackNetV3,
    onnx_path: Path,
    *,
    input_width: int,
    input_height: int,
    opset: int,
    device: str,
) -> None:
    try:
        import onnx  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "导出 ONNX 需要安装 onnx 包。请在当前 Python 环境执行: python -m pip install onnx"
        ) from exc

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 9, input_height, input_width, device=device)
    print(f"[export] ONNX -> {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["heatmap"],
        dynamic_axes={
            "input": {0: "batch"},
            "heatmap": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )


def collect_media_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Calibration source not found: {source}")

    paths = [
        p
        for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS)
    ]
    return sorted(paths)


def preprocess_window(frames: list[np.ndarray], input_width: int, input_height: int) -> np.ndarray:
    if len(frames) != 3:
        raise ValueError("TrackNet calibration expects exactly 3 frames per window.")
    stacked = np.empty((9, input_height, input_width), dtype=np.uint8)
    for index, frame in enumerate(frames):
        resized = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        stacked[index * 3 : (index + 1) * 3] = rgb.transpose(2, 0, 1)
    return stacked.astype(np.float32) * (1.0 / 255.0)


def iter_image_windows(paths: list[Path]) -> Iterator[list[np.ndarray]]:
    frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        return
    if len(frames) == 1:
        yield [frames[0], frames[0], frames[0]]
        return
    if len(frames) == 2:
        yield [frames[0], frames[1], frames[1]]
        return
    for index in range(1, len(frames) - 1):
        yield [frames[index - 1], frames[index], frames[index + 1]]


def iter_video_windows(path: Path, stride: int, max_seconds: float) -> Iterator[list[np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[warn] skip unreadable video: {path}")
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    max_frame_count = int(round(max_seconds * fps)) if max_seconds > 0 else 0

    previous: np.ndarray | None = None
    current: np.ndarray | None = None
    frame_index = 0
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return
        previous = frame
        ok, frame = cap.read()
        if not ok or frame is None:
            yield [previous, previous, previous]
            return
        current = frame

        while True:
            if max_frame_count > 0 and frame_index >= max_frame_count:
                break
            ok, next_frame = cap.read()
            if not ok or next_frame is None:
                break
            if frame_index % max(1, stride) == 0:
                yield [previous, current, next_frame]
            previous, current = current, next_frame
            frame_index += 1
    finally:
        cap.release()


def iter_calibration_samples(
    source: Path,
    *,
    input_width: int,
    input_height: int,
    stride: int,
    max_seconds: float,
) -> Iterator[np.ndarray]:
    paths = collect_media_paths(source)
    if not paths:
        raise RuntimeError(f"No calibration videos/images found in: {source}")

    image_paths = [path for path in paths if path.suffix.lower() in IMAGE_EXTS]
    video_paths = [path for path in paths if path.suffix.lower() in VIDEO_EXTS]

    for video_path in video_paths:
        limit_text = f" | limit {max_seconds:.1f}s" if max_seconds > 0 else ""
        print(f"[calib] video: {video_path}{limit_text}")
        for frames in iter_video_windows(video_path, stride, max_seconds):
            yield preprocess_window(frames, input_width, input_height)

    if image_paths:
        print(f"[calib] images: {len(image_paths)}")
        for frames in iter_image_windows(image_paths):
            yield preprocess_window(frames, input_width, input_height)


def iter_calibration_batches(
    samples: Iterable[np.ndarray],
    *,
    batch_size: int,
    max_batches: int,
) -> Iterator[np.ndarray]:
    batch: list[np.ndarray] = []
    produced = 0
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield np.stack(batch, axis=0)
            produced += 1
            batch.clear()
            if produced >= max_batches:
                return
    if batch and produced < max_batches:
        while len(batch) < batch_size:
            batch.append(batch[-1])
        yield np.stack(batch, axis=0)


def make_calibrator(
    trt,
    *,
    calib_source: Path,
    input_width: int,
    input_height: int,
    batch_size: int,
    max_batches: int,
    stride: int,
    max_seconds: float,
    cache_path: Path,
):
    class TorchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            samples = iter_calibration_samples(
                calib_source,
                input_width=input_width,
                input_height=input_height,
                stride=stride,
                max_seconds=max_seconds,
            )
            self._batches = iter_calibration_batches(
                samples,
                batch_size=batch_size,
                max_batches=max_batches,
            )
            self._current: torch.Tensor | None = None

        def get_batch_size(self) -> int:
            return batch_size

        def get_batch(self, names) -> list[int] | None:
            try:
                batch = next(self._batches)
            except StopIteration:
                return None
            self._current = torch.from_numpy(batch).cuda(non_blocking=True).contiguous()
            return [int(self._current.data_ptr())]

        def read_calibration_cache(self):
            if cache_path.is_file():
                print(f"[calib] using cache: {cache_path}")
                return cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache) -> None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(bytes(cache))
            print(f"[calib] wrote cache: {cache_path}")

    return TorchEntropyCalibrator()


def set_workspace(config, trt, workspace_gb: float) -> None:
    workspace_bytes = int(max(workspace_gb, 0.25) * (1024**3))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def build_engine(args: argparse.Namespace, onnx_path: Path, output_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build and run a TensorRT engine.")

    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install TensorRT Python package in this environment first.") from exc

    logger_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
    logger = trt.Logger(logger_level)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    print(f"[build] parsing ONNX: {onnx_path}")
    if not parser.parse(onnx_path.read_bytes()):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        raise RuntimeError("TensorRT failed to parse ONNX.")

    config = builder.create_builder_config()
    set_workspace(config, trt, args.workspace_gb)

    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 tactics enabled")

    if not builder.platform_has_fast_int8:
        print("[warn] builder reports no fast INT8 support; engine will still be built with INT8 flag if possible.")
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = make_calibrator(
        trt,
        calib_source=Path(args.calib_source),
        input_width=args.input_width,
        input_height=args.input_height,
        batch_size=args.calib_batch_size,
        max_batches=args.calib_batches,
        stride=args.calib_stride,
        max_seconds=args.calib_seconds,
        cache_path=output_path.with_suffix(".calib.cache"),
    )

    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    profile = builder.create_optimization_profile()
    min_shape = (args.min_batch, 9, args.input_height, args.input_width)
    opt_shape = (args.opt_batch, 9, args.input_height, args.input_width)
    max_shape = (args.max_batch, 9, args.input_height, args.input_width)
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    print(f"[build] profile {input_name}: min={min_shape} opt={opt_shape} max={max_shape}")

    print(f"[build] TensorRT INT8 engine -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed.")
    output_path.write_bytes(bytes(serialized))
    print(f"[done] wrote: {output_path}")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    onnx_path = Path(args.onnx).resolve() if args.onnx else output_path.with_suffix(".onnx")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device}")
    model = load_tracknet(checkpoint_path, device)
    export_onnx(
        model,
        onnx_path,
        input_width=args.input_width,
        input_height=args.input_height,
        opset=args.opset,
        device=device,
    )
    build_engine(args, onnx_path, output_path)


if __name__ == "__main__":
    main()
