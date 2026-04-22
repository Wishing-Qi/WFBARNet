from __future__ import annotations

import ctypes
import sys
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import torch

from apps.desktop_gui.controllers.tracknet_controller import TrackTaskConfig, run_tracknet_task
from apps.desktop_gui.state.gui_state import gui_state
from apps.desktop_gui.utils.theme import apply_global_theme


WINDOW_TITLE = "羽毛球轨迹分析工作台 - Advanced Vision AI"
PLACEHOLDER_VIDEO_SIZE = (730, 390)
VIDEO_PANEL_HEIGHT = 552
LOG_PANEL_HEIGHT = 240  # 修复：原代码缺失该高度常量
TACTICAL_MAP_SIZE = (240, 320)
EVENT_QUEUE: Queue[dict[str, Any]] = Queue()

# 现代状态调色板 (Tailwind 系)
STATUS_COLORS = {
    "idle": (148, 163, 184),          # Gray
    "video_selected": (56, 189, 248), # Light Blue
    "running": (16, 185, 129),        # Emerald Green
    "completed": (34, 197, 94),       # Bright Green
    "failed": (248, 113, 113),        # Rose Red
    "stopped": (245, 158, 11),        # Amber Orange
}

# UI 点缀强调色
ACCENT_TEAL = (45, 212, 191)
ACCENT_CYAN = (103, 232, 249)
ACCENT_BLUE = (59, 130, 246)
ACCENT_PURPLE = (167, 139, 250)
ACCENT_GREEN = (34, 197, 94)
ACCENT_ORANGE = (251, 191, 36)
ACCENT_MUTED = (148, 163, 184)


def get_dpi_scale() -> float:
    if sys.platform != "win32":
        return 1.0
    try:
        user32 = ctypes.windll.user32
        dpi = user32.GetDpiForSystem()
        return max(1.0, float(dpi) / 96.0) if dpi else 1.0
    except Exception:
        return 1.0


def setup_fonts() -> None:
    scale = get_dpi_scale()
    font_size = max(15, round(15 * scale))
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is None:
        return

    with dpg.font_registry():
        with dpg.font(str(font_path), font_size) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
        dpg.bind_font(default_font)


def available_devices() -> list[str]:
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        devices.insert(1, "cuda:0")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def append_log(level: str, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    gui_state.logs.append(f"[{timestamp}] [{level}] {message}")
    _refresh_logs()


def _noop_callback(*_args: object, **_kwargs: object) -> None:
    return


def _empty_texture_data() -> list[float]:
    """生成更具科技感的高级相机占位画面"""
    width, height = PLACEHOLDER_VIDEO_SIZE
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (12, 14, 18)  # 深邃背板

    # 细致的十字准星网格
    grid_color = (24, 28, 36)
    for x in range(0, width, 40):
        cv2.line(canvas, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, 40):
        cv2.line(canvas, (0, y), (width, y), grid_color, 1)

    # 中心准星标
    cx, cy = width // 2, height // 2 - 20
    icon_color = (80, 85, 95)
    cv2.line(canvas, (cx - 40, cy), (cx + 40, cy), icon_color, 1)
    cv2.line(canvas, (cx, cy - 40), (cx, cy + 40), icon_color, 1)
    cv2.circle(canvas, (cx, cy), 15, icon_color, 1)

    text = "NO SIGNAL / AWAITING TENSOR"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
    cv2.putText(
        canvas,
        text,
        ((width - text_size[0]) // 2, cy + 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (100, 105, 120),
        1,
        lineType=cv2.LINE_AA,
    )

    rgba = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgba).ravel().tolist()


def _frame_to_texture_data(frame: np.ndarray) -> list[float]:
    width, height = PLACEHOLDER_VIDEO_SIZE
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgba).ravel().tolist()


def _update_preview(frame: np.ndarray | None) -> None:
    if not dpg.does_item_exist("video_texture"):
        return
    dpg.set_value("video_texture", _frame_to_texture_data(frame) if frame is not None else _empty_texture_data())


def _format_timecode(frame_idx: int, fps: float) -> str:
    if fps <= 0:
        return "00:00.00"
    total_seconds = frame_idx / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60
    return f"{minutes:02d}:{seconds:05.2f}"


def _estimate_metric_coordinates(ball_xy: list[float], frame_shape: tuple[int, int] | None) -> tuple[float, float]:
    if frame_shape is None:
        return 0.0, 0.0
    frame_height, frame_width = frame_shape
    if frame_width <= 0 or frame_height <= 0:
        return 0.0, 0.0
    x = ((ball_xy[0] / frame_width) - 0.5) * 6.1
    y = (0.5 - (ball_xy[1] / frame_height)) * 13.4
    return x, y


def _map_track_point_to_tactical(ball_xy: list[float], frame_shape: tuple[int, int] | None) -> tuple[float, float]:
    if frame_shape is None:
        return 0.5, 0.5
    frame_height, frame_width = frame_shape
    if frame_width <= 0 or frame_height <= 0:
        return 0.5, 0.5
    x = min(max(ball_xy[0] / frame_width, 0.0), 1.0)
    y = min(max(ball_xy[1] / frame_height, 0.0), 1.0)
    return x, y


def _refresh_tactical_map() -> None:
    """带有发光特效 (Glow Effect) 的现代化战术画板绘制"""
    if not dpg.does_item_exist("tactical_map_drawlist"):
        return

    dpg.delete_item("tactical_map_drawlist", children_only=True)
    width, height = TACTICAL_MAP_SIZE
    pad_x = 64
    pad_y = 18
    court_left = pad_x
    court_top = pad_y
    court_right = width - pad_x
    court_bottom = height - pad_y
    court_width = court_right - court_left
    court_height = court_bottom - court_top

    # 背景与场地边缘
    dpg.draw_rectangle((0, 0), (width, height), color=(45, 48, 60), fill=(24, 24, 27), rounding=8, parent="tactical_map_drawlist")
    dpg.draw_rectangle(
        (court_left, court_top),
        (court_right, court_bottom),
        color=(63, 63, 70),
        fill=(6, 78, 59), # 深体育场绿
        thickness=2,
        parent="tactical_map_drawlist",
    )

    # 场地标线
    line_color = (255, 255, 255, 120)  # 半透明白色，更显现代
    for offset in (0.25, 0.5, 0.75):
        y = court_top + court_height * offset
        dpg.draw_line((court_left, y), (court_right, y), color=line_color, thickness=1, parent="tactical_map_drawlist")
    
    dpg.draw_line(
        (court_left + court_width * 0.5, court_top),
        (court_left + court_width * 0.5, court_bottom),
        color=line_color,
        thickness=1,
        parent="tactical_map_drawlist",
    )
    
    # 球网
    dpg.draw_line(
        (court_left, court_top + court_height * 0.5),
        (court_right, court_top + court_height * 0.5),
        color=(203, 213, 225),
        thickness=3,
        parent="tactical_map_drawlist",
    )

    frame_shape = None
    if gui_state.current_frame_image is not None:
        frame_shape = gui_state.current_frame_image.shape[:2]

    # 获取有效轨迹点
    visible_points: list[tuple[float, float]] = []
    for frame_idx in sorted(gui_state.track_results.keys()):
        item = gui_state.track_results[frame_idx]
        if not item.get("visible"):
            continue
        nx, ny = _map_track_point_to_tactical(item.get("ball_xy", [0.0, 0.0]), frame_shape)
        visible_points.append(
            (court_left + nx * court_width, court_top + ny * court_height)
        )

    visible_points = visible_points[-25:] # 增加尾迹长度

    # 渲染历史轨迹线
    if len(visible_points) >= 2:
        dpg.draw_polyline(visible_points, color=(251, 191, 36, 180), thickness=2, parent="tactical_map_drawlist")

    # 渲染历史球点
    if visible_points:
        for i, point in enumerate(visible_points[:-1]):
            alpha = int(50 + (i / len(visible_points)) * 150)  # 渐变透明度尾迹
            dpg.draw_circle(point, 2.0, fill=(251, 146, 60, alpha), color=(251, 146, 60, alpha), parent="tactical_map_drawlist")
        
        # 渲染当前羽毛球点 (带光晕特效)
        current_pt = visible_points[-1]
        dpg.draw_circle(current_pt, 12.0, fill=(251, 191, 36, 30), color=(0,0,0,0), parent="tactical_map_drawlist") # 大外发光
        dpg.draw_circle(current_pt, 6.0, fill=(251, 191, 36, 80), color=(0,0,0,0), parent="tactical_map_drawlist")  # 内发光
        dpg.draw_circle(current_pt, 3.5, fill=(255, 255, 255), color=(251, 191, 36), parent="tactical_map_drawlist") # 核心


def _load_video_preview(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件：{video_path}")

    ok, frame = cap.read()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"视频已打开，但没有成功读取到首帧：{video_path}")

    gui_state.current_video_path = video_path
    gui_state.current_frame_image = frame
    gui_state.current_fps = fps
    gui_state.total_frames = total_frames
    gui_state.current_frame_idx = 0
    gui_state.summary.clear()
    gui_state.actions.clear()
    gui_state.track_results.clear()
    gui_state.result_files.clear()
    gui_state.status = "video_selected"
    gui_state.task_progress = 0.0
    gui_state.current_stage = "视频已加载，等待开始分析"
    gui_state.error_message = ""
    _update_preview(frame)


def _build_output_dir() -> Path:
    video_stem = Path(gui_state.current_video_path or "tracknet").stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(gui_state.output_dir) / f"{video_stem}_{timestamp}"


def _run_worker(config: TrackTaskConfig) -> None:
    try:
        result = run_tracknet_task(
            config,
            progress_callback=lambda payload: EVENT_QUEUE.put(payload),
            stop_requested=lambda: gui_state.stop_requested,
        )
        EVENT_QUEUE.put({"type": "complete", "payload": result})
    except Exception as exc:
        EVENT_QUEUE.put({"type": "error", "message": str(exc)})


# ==========================================
# 交互回调
# ==========================================

def on_open_video_dialog() -> None:
    dpg.show_item("video_file_dialog")


def on_video_selected(_sender: str, app_data: dict[str, Any], _user_data: object) -> None:
    file_path = app_data.get("file_path_name", "")
    if not file_path:
        return
    try:
        _load_video_preview(file_path)
        append_log("INFO", f"已载入视频：{file_path}")
    except Exception as exc:
        gui_state.status = "failed"
        gui_state.current_stage = "视频加载失败"
        gui_state.error_message = str(exc)
        append_log("ERROR", str(exc))
        _update_preview(None)
    refresh_ui_from_state()
    _refresh_timeline()
    _refresh_output_files()


def on_device_changed(_sender: str, value: str) -> None:
    gui_state.device = value


def on_score_threshold_changed(_sender: str, value: float) -> None:
    gui_state.track_score_threshold = float(value)


def on_max_frames_changed(_sender: str, value: int) -> None:
    gui_state.max_frames = max(0, int(value))


def on_save_visualization_changed(_sender: str, value: bool) -> None:
    gui_state.save_visualization = bool(value)


def on_save_json_changed(_sender: str, value: bool) -> None:
    gui_state.save_json = bool(value)


def on_save_csv_changed(_sender: str, value: bool) -> None:
    gui_state.save_csv = bool(value)


def on_save_npy_changed(_sender: str, value: bool) -> None:
    gui_state.save_npy = bool(value)


def on_start() -> None:
    if gui_state.worker is not None and gui_state.worker.is_alive():
        append_log("WARNING", "当前任务仍在运行，请等待完成或先停止。")
        return
    if not gui_state.current_video_path:
        append_log("WARNING", "请先选择一个本地视频文件。")
        return

    gui_state.clear_results(keep_video=True)
    gui_state.stop_requested = False
    gui_state.status = "running"
    gui_state.current_stage = "正在初始化 TrackNetV3 推理引擎..."
    gui_state.task_progress = 0.0
    gui_state.error_message = ""

    output_dir = _build_output_dir()
    config = TrackTaskConfig(
        source=gui_state.current_video_path,
        output_dir=output_dir,
        device=gui_state.device,
        score_threshold=gui_state.track_score_threshold,
        max_frames=gui_state.max_frames if gui_state.max_frames > 0 else None,
        save_visualization=gui_state.save_visualization,
        save_json=gui_state.save_json,
        save_csv=gui_state.save_csv,
        save_npy=gui_state.save_npy,
    )
    append_log("EXEC", f"启动 TrackNetV3 推理任务，输出目录：{output_dir}")

    worker = Thread(target=_run_worker, args=(config,), daemon=True)
    gui_state.worker = worker
    worker.start()
    refresh_ui_from_state()


def on_stop() -> None:
    if gui_state.worker is None or not gui_state.worker.is_alive():
        append_log("WARNING", "当前没有正在运行的任务。")
        return
    gui_state.stop_requested = True
    gui_state.current_stage = "正在停止，请等待当前帧处理完成..."
    append_log("WARNING", "已请求停止当前任务。")
    refresh_ui_from_state()


def on_reset() -> None:
    if gui_state.worker is not None and gui_state.worker.is_alive():
        gui_state.stop_requested = True
        append_log("WARNING", "任务仍在运行，已先请求停止。请稍候再次点击重置。")
        return
    gui_state.reset_runtime_fields()
    _update_preview(None)
    _refresh_timeline()
    _refresh_output_files()
    refresh_ui_from_state()
    _refresh_logs()
    append_log("INFO", "界面已重置。")


# ==========================================
# 视图构建逻辑
# ==========================================

def create_sidebar_panel() -> None:
    with dpg.child_window(tag="sidebar_panel", border=False, height=-1):
        dpg.add_text("🎛️ 控制台配置", color=(236, 242, 252))
        dpg.add_spacer(height=10)

        with dpg.collapsing_header(label="📁 输入与模型", default_open=True):
            dpg.add_spacer(height=2)
            dpg.add_text("视频源文件", color=(170, 180, 200))
            with dpg.group(horizontal=True):
                video_path_input = dpg.add_input_text(
                    hint="请选择视频文件...",
                    tag="input_video_path",
                    readonly=True,
                    width=-50,
                )
                browse_btn = dpg.add_button(label="...", width=40, height=30, callback=on_open_video_dialog)
                dpg.bind_item_theme(browse_btn, "theme_button_subtle")
                dpg.bind_item_theme(video_path_input, "theme_input")

            dpg.add_spacer(height=6)
            dpg.add_text("推理流水线 (Pipeline)", color=(170, 180, 200))
            pipeline_combo = dpg.add_combo(
                items=["TrackNetV3 (Unified)", "Pose Only", "Track Realtime"],
                default_value="TrackNetV3 (Unified)",
                width=-1,
                enabled=False,
            )
            dpg.bind_item_theme(pipeline_combo, "theme_input")
            dpg.add_spacer(height=8)

        with dpg.collapsing_header(label="⚙️ 引擎参数", default_open=True):
            dpg.add_spacer(height=2)
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("计算设备", color=(170, 180, 200))
                    device_combo = dpg.add_combo(
                        items=available_devices(),
                        default_value=gui_state.device,
                        tag="device_combo",
                        width=120,
                        callback=on_device_changed,
                    )
                    dpg.bind_item_theme(device_combo, "theme_input")
                with dpg.group():
                    dpg.add_text("半精度 (FP16)", color=(170, 180, 200))
                    fp16_combo = dpg.add_combo(
                        items=["Enabled", "Disabled"],
                        default_value="Enabled" if gui_state.device != "cpu" else "Disabled",
                        width=120,
                        enabled=False,
                    )
                    dpg.bind_item_theme(fp16_combo, "theme_input")
            dpg.add_spacer(height=8)

        with dpg.collapsing_header(label="📤 导出配置", default_open=True):
            dpg.add_spacer(height=2)
            checkbox_vis = dpg.add_checkbox(
                label="渲染可视化视频 (MP4)",
                default_value=gui_state.save_visualization,
                callback=on_save_visualization_changed,
            )
            checkbox_json = dpg.add_checkbox(
                label="导出背景与轨迹 (JSON)",
                default_value=gui_state.save_json,
                callback=on_save_json_changed,
            )
            checkbox_npy = dpg.add_checkbox(
                label="生成动作识别特征 (NPY)",
                default_value=gui_state.save_npy,
                callback=on_save_npy_changed,
            )
            dpg.bind_item_theme(checkbox_vis, "theme_checkbox")
            dpg.bind_item_theme(checkbox_json, "theme_checkbox")
            dpg.bind_item_theme(checkbox_npy, "theme_checkbox")
            dpg.add_spacer(height=8)

        dpg.add_spacer(height=20)
        dpg.add_text("PROCESS CONTROL", color=(112, 125, 148))
        dpg.add_separator()
        dpg.add_spacer(height=8)
        
        # 按钮组对齐排版
        with dpg.group(horizontal=True):
            start_btn = dpg.add_button(label="▶ 开始分析", tag="btn_start", width=140, height=38, callback=on_start)
            stop_btn = dpg.add_button(label="⏹ 停止", tag="btn_stop", width=70, height=38, callback=on_stop)
            reset_btn = dpg.add_button(label="⟳ 重置", tag="btn_reset", width=70, height=38, callback=on_reset)
            
            dpg.bind_item_theme(start_btn, "theme_button_success")
            dpg.bind_item_theme(stop_btn, "theme_button_danger")
            dpg.bind_item_theme(reset_btn, "theme_button_subtle")

        dpg.add_spacer(height=16)
        with dpg.group(horizontal=True):
            dpg.add_text("● IDLE", tag="text_status", color=STATUS_COLORS["idle"])
            dpg.add_spacer(width=10)
            dpg.add_text("Stage: 等待载入视频", tag="text_stage", color=(160, 175, 205))
        dpg.add_spacer(height=6)
        progress_bar = dpg.add_progress_bar(tag="task_progress", default_value=0.0, width=-1)
        dpg.bind_item_theme(progress_bar, "theme_progress_bar")


def create_video_panel() -> None:
    width, height = PLACEHOLDER_VIDEO_SIZE
    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(
            width,
            height,
            _empty_texture_data(),
            tag="video_texture",
        )

    with dpg.child_window(tag="video_panel", border=False, height=VIDEO_PANEL_HEIGHT, no_scrollbar=True):
        with dpg.child_window(tag="viewport_shell", border=True, height=-1, no_scrollbar=True):
            with dpg.table(
                header_row=False, borders_innerH=False, borders_innerV=False, borders_outerH=False, borders_outerV=False,
            ):
                dpg.add_table_column(width_stretch=True)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=90)
                with dpg.table_row():
                    dpg.add_text("🎥 视频检测器 (Viewport)", color=(228, 233, 246))
                    dpg.add_text("FPS: 0.0", tag="video_fps", color=ACCENT_GREEN)

            dpg.add_spacer(height=10)
            with dpg.child_window(tag="viewport_canvas_card", border=True, height=height + 16, no_scrollbar=True):
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=4)
                    dpg.add_image(
                        "video_texture",
                        tag="video_image_item",
                        width=PLACEHOLDER_VIDEO_SIZE[0],
                        height=PLACEHOLDER_VIDEO_SIZE[1],
                    )

            dpg.add_spacer(height=12)
            with dpg.child_window(tag="viewport_transport", border=True, height=64, no_scrollbar=True):
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=2)
                    pause_btn = dpg.add_button(label="⏸", width=36, height=32, callback=_noop_callback)
                    play_btn = dpg.add_button(label="▶", width=36, height=32, callback=_noop_callback)
                    dpg.bind_item_theme(pause_btn, "theme_button_primary")
                    dpg.bind_item_theme(play_btn, "theme_button_subtle")
                    
                    dpg.add_spacer(width=10)
                    playback_slider = dpg.add_slider_int(
                        tag="playback_slider", default_value=0, min_value=0, max_value=1, width=-150, enabled=False, format=""
                    )
                    dpg.bind_item_theme(playback_slider, "theme_slider")
                    
                    dpg.add_spacer(width=10)
                    dpg.add_text("00:00 / 00:00", tag="video_timecode", color=(208, 214, 230))
                
        dpg.bind_item_theme("viewport_shell", "theme_card")
        dpg.bind_item_theme("viewport_canvas_card", "theme_card_soft")
        dpg.bind_item_theme("viewport_transport", "theme_card_soft")


def _add_info_row(label: str, tag: str, default_value: str, accent_color: tuple = ACCENT_PURPLE) -> None:
    with dpg.table_row():
        dpg.add_text(label, color=(160, 175, 205))
        dpg.add_text(default_value, tag=tag, color=accent_color)


def create_info_panel() -> None:
    with dpg.child_window(tag="info_panel", border=False, height=-1, no_scrollbar=True):
        dpg.add_text("🎯 实时探针 (Inspector)", color=(228, 233, 246))
        dpg.add_spacer(height=10)

        with dpg.child_window(tag="inspector_metrics_card", border=True, height=104, no_scrollbar=True):
            with dpg.table(
                header_row=False, borders_innerH=True, borders_innerV=False, borders_outerH=False, borders_outerV=False,
            ):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=106)
                dpg.add_table_column(width_stretch=True)
                _add_info_row("逆透视坐标", "inspector_metric_coords", "X: -, Y: -", ACCENT_TEAL)
                _add_info_row("轨迹置信度", "inspector_metric_score", "0.000", ACCENT_GREEN)

        dpg.add_spacer(height=16)
        with dpg.group(horizontal=True):
            dpg.add_text("📊 TACTICAL MAP", color=(182, 188, 205))
            dpg.add_spacer(width=30)
            dpg.add_text("● Player A", color=(255, 91, 91))
            dpg.add_text("● Player B", color=(83, 150, 255))

        dpg.add_spacer(height=6)
        with dpg.child_window(tag="tactical_map_card", border=True, height=TACTICAL_MAP_SIZE[1] + 28, no_scrollbar=True):
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=28)
                dpg.add_spacer(height=14) # top padding for canvas inside card
                dpg.add_drawlist(width=TACTICAL_MAP_SIZE[0], height=TACTICAL_MAP_SIZE[1], tag="tactical_map_drawlist")

        dpg.bind_item_theme("inspector_metrics_card", "theme_card")
        dpg.bind_item_theme("tactical_map_card", "theme_card")


def create_log_panel() -> None:
    with dpg.child_window(tag="log_panel", border=False, height=-1, no_scrollbar=True):
        with dpg.child_window(tag="log_shell", border=True, height=-1):
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=4)
                dpg.add_text("📜 执行日志 (Console)", color=(228, 233, 246))
            dpg.add_separator()
            with dpg.child_window(tag="log_scroll_area", width=-1, height=-1, border=False):
                pass

        dpg.bind_item_theme("log_shell", "theme_card_soft")


def create_file_dialog() -> None:
    with dpg.file_dialog(
        directory_selector=False, show=False, callback=on_video_selected, tag="video_file_dialog",
        width=760, height=460, modal=True,
    ):
        dpg.add_file_extension("视频文件 (*.mp4 *.avi *.mov *.mkv){.mp4,.avi,.mov,.mkv}", color=(96, 165, 250, 255))
        dpg.add_file_extension(".*")


def _log_color(message: str) -> tuple[int, int, int]:
    if "[ERROR]" in message:
        return (248, 113, 113)
    if "[WARNING]" in message:
        return (251, 191, 36)
    if "[EXEC]" in message:
        return (45, 212, 191)
    if "[DEBUG]" in message:
        return (167, 139, 250)
    if "[INFO]" in message:
        return (96, 165, 250)
    if "[SUCCESS]" in message or "完成" in message:
        return (34, 197, 94)
    return (200, 210, 225)


def _refresh_logs() -> None:
    if not dpg.does_item_exist("log_scroll_area"):
        return
    dpg.delete_item("log_scroll_area", children_only=True)
    for line in gui_state.logs[-240:]:
        dpg.add_text(line, parent="log_scroll_area", color=_log_color(line), wrap=640)
    dpg.set_y_scroll("log_scroll_area", dpg.get_y_scroll_max("log_scroll_area"))


def _refresh_timeline() -> None:
    pass # 留给摘要视图的占位，视需要后续补充


def _refresh_output_files() -> None:
    pass


def _live_track_stats() -> tuple[int, int, float, float, float]:
    track_items = list(gui_state.track_results.values())
    total = len(track_items)
    visible = sum(1 for item in track_items if item.get("visible"))
    scores = [float(item.get("score", 0.0)) for item in track_items]
    visibility_ratio = visible / total if total else 0.0
    avg_conf = sum(scores) / total if total else 0.0
    peak_conf = max(scores, default=0.0)
    return total, visible, visibility_ratio, avg_conf, peak_conf


def _status_badge(status: str) -> str:
    return {
        "idle": "IDLE", "video_selected": "READY", "running": "RUNNING",
        "completed": "COMPLETED", "failed": "ERROR", "stopped": "STOPPED",
    }.get(status, status.upper())


def drain_worker_events() -> None:
    while True:
        try:
            payload = EVENT_QUEUE.get_nowait()
        except Empty:
            break

        event_type = payload.get("type")
        if event_type == "stage":
            next_stage = str(payload.get("stage", gui_state.current_stage))
            if next_stage and next_stage != gui_state.current_stage:
                append_log("INFO", next_stage)
            gui_state.current_stage = next_stage
            gui_state.task_progress = max(gui_state.task_progress, float(payload.get("progress", 0.0)))
        elif event_type == "frame":
            gui_state.current_stage = str(payload.get("stage", gui_state.current_stage))
            gui_state.current_frame_idx = int(payload.get("frame_id", gui_state.current_frame_idx))
            gui_state.task_progress = float(payload.get("progress", gui_state.task_progress))
            frame = payload.get("frame")
            if isinstance(frame, np.ndarray):
                gui_state.current_frame_image = frame
                _update_preview(frame)
            track = payload.get("track")
            if isinstance(track, dict):
                gui_state.track_results[gui_state.current_frame_idx] = track
                _refresh_tactical_map()
            processed = len(gui_state.track_results)
            if processed in {1, 30} or (processed > 0 and processed % 120 == 0):
                append_log("DEBUG", f"已处理 {processed} 帧，当前阶段：{gui_state.current_stage}")
        elif event_type == "complete":
            result = payload["payload"]
            gui_state.worker = None
            gui_state.summary = result.summary
            gui_state.actions = result.actions
            gui_state.track_results = result.track_results
            gui_state.result_files = result.output_files
            gui_state.current_fps = result.fps
            gui_state.total_frames = result.total_frames
            gui_state.current_frame_idx = result.last_frame_idx
            gui_state.task_progress = result.progress
            gui_state.status = "stopped" if result.stopped else "completed"
            gui_state.current_stage = "任务已停止，已保留当前结果" if result.stopped else "推理执行完成"
            if result.last_frame is not None:
                gui_state.current_frame_image = result.last_frame
                _update_preview(result.last_frame)
            _refresh_tactical_map()
            append_log("WARNING" if result.stopped else "SUCCESS", gui_state.current_stage)
        elif event_type == "error":
            gui_state.worker = None
            gui_state.status = "failed"
            gui_state.current_stage = "推理执行失败"
            gui_state.error_message = str(payload.get("message", "未知错误"))
            append_log("ERROR", gui_state.error_message)


def refresh_ui_from_state() -> None:
    if dpg.does_item_exist("task_progress"):
        dpg.set_value("task_progress", gui_state.task_progress)

    if dpg.does_item_exist("text_status"):
        dpg.set_value("text_status", f"● {_status_badge(gui_state.status)}")
        dpg.configure_item("text_status", color=STATUS_COLORS.get(gui_state.status, (225, 230, 238)))

    if dpg.does_item_exist("text_stage"):
        dpg.set_value("text_stage", f"Stage: {gui_state.current_stage or '-'}")

    if dpg.does_item_exist("input_video_path"):
        dpg.set_value("input_video_path", gui_state.current_video_path or "")

    display_current = gui_state.current_frame_idx + 1 if (gui_state.total_frames or gui_state.track_results) else 0
    display_total = gui_state.total_frames or max(len(gui_state.track_results), display_current)

    if dpg.does_item_exist("video_meta"):
        dpg.set_value("video_meta", f"Frame {display_current} / {display_total}")

    if dpg.does_item_exist("video_fps"):
        dpg.set_value("video_fps", f"FPS: {gui_state.current_fps:.1f}")

    if dpg.does_item_exist("video_timecode"):
        current_time = _format_timecode(gui_state.current_frame_idx, gui_state.current_fps)
        total_time = _format_timecode(display_total, gui_state.current_fps)
        dpg.set_value("video_timecode", f"{current_time} / {total_time}")

    if dpg.does_item_exist("playback_slider"):
        slider_max = max(display_total, 1)
        dpg.configure_item("playback_slider", max_value=slider_max)
        dpg.set_value("playback_slider", min(display_current, slider_max))

    track = gui_state.track_results.get(gui_state.current_frame_idx, {})
    ball_xy = track.get("ball_xy", [-1, -1])
    visible = bool(track.get("visible", False))
    score = float(track.get("score", 0.0))
    frame_shape = gui_state.current_frame_image.shape[:2] if gui_state.current_frame_image is not None else None
    metric_x, metric_y = _estimate_metric_coordinates(ball_xy, frame_shape) if visible else (0.0, 0.0)

    if dpg.does_item_exist("inspector_metric_coords"):
        dpg.set_value("inspector_metric_coords", f"X: {metric_x:+.1f}m, Y: {metric_y:+.1f}m" if visible else "X: -, Y: -")
    if dpg.does_item_exist("inspector_metric_score"):
        dpg.set_value("inspector_metric_score", f"{score:.3f}")


def build_gui() -> None:
    dpg.create_context()
    apply_global_theme()
    setup_fonts()
    create_file_dialog()

    with dpg.window(tag="PrimaryWindow", no_scrollbar=True, no_scroll_with_mouse=True):
        with dpg.table(
            header_row=False, borders_innerH=False, borders_innerV=True, borders_outerH=False, borders_outerV=False,
        ):
            dpg.add_table_column(width_fixed=True, init_width_or_weight=360)
            dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
            dpg.add_table_column(width_fixed=True, init_width_or_weight=340)

            with dpg.table_row():
                create_sidebar_panel()
                with dpg.child_window(tag="center_panel", border=False, height=-1, no_scrollbar=True):
                    create_video_panel()
                    dpg.add_spacer(height=12)
                    create_log_panel()
                create_info_panel()

    # 绑定统一卡片主题
    dpg.bind_item_theme("sidebar_panel", "theme_card")
    dpg.bind_item_theme("info_panel", "theme_card")

    dpg.create_viewport(title=WINDOW_TITLE, width=1540, height=930, min_width=1420, min_height=820)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("PrimaryWindow", True)

    _update_preview(gui_state.current_frame_image)
    _refresh_logs()
    _refresh_tactical_map()
    refresh_ui_from_state()

    while dpg.is_dearpygui_running():
        drain_worker_events()
        refresh_ui_from_state()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()

if __name__ == "__main__":
    build_gui()