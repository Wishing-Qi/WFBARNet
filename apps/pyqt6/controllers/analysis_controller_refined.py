from __future__ import annotations

from pathlib import Path
from threading import Thread

import cv2
import numpy as np
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFileDialog
from PyQt6.sip import isdeleted

from apps.pyqt6.utils.style import apply_theme, discover_themes
from apps.pyqt6.views.main_window_refined import MainWindow


class ProbeWorker(QThread):
    """后台探测视频元数据，避免主线程阻塞。"""
    finished = pyqtSignal(str, dict)

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self._file_path = file_path

    def run(self) -> None:
        cap = cv2.VideoCapture(self._file_path)
        if not cap.isOpened():
            self.finished.emit(self._file_path, {"fps": 0.0, "width": 0, "height": 0})
            return
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        self.finished.emit(self._file_path, {"fps": fps, "width": width, "height": height})


class MainController:
    """控制层：连接视频选择、TrackNet 推理和结果显示。"""

    def __init__(self, view: MainWindow) -> None:
        self.view = view
        self._selected_video_path: str | None = None
        self._probe_worker: ProbeWorker | None = None
        self._theme_dirs = discover_themes()
        self._track_branch = None
        self._track_running = False

        self._init_tracknet()
        self._bind_events()
        self.view.populate_stylesheets(self._theme_dirs)
        self._set_idle_state()
        self.view.append_log(f"[System] 界面已初始化")

    def _init_tracknet(self) -> None:
        from src.models.track_branch import TrackBranch
        project_root = Path(__file__).resolve().parents[3]
        model_weight = str(project_root / "assets" / "weights" / "track" / "model_best.pt")
        self._track_branch = TrackBranch(
            model_weight=model_weight,
            device="cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
            input_size=(512, 288),
            score_thr=0.35,
        )
        self.view.append_log(f"[TrackNet] 模型已加载")

    def _bind_events(self) -> None:
        self.view.video_player.selectRequested.connect(self.handle_upload)
        self.view.video_player.forceStopRequested.connect(self.handle_force_stop)
        self.view._style_menu.triggered.connect(self._on_style_action_triggered)

    def _set_idle_state(self) -> None:
        self.view.video_player.btn_select_video.setEnabled(True)
        self.view.video_player.btn_force_stop.setEnabled(True)
        self.view.set_status_state("idle")

    def _set_running_state(self) -> None:
        self.view.video_player.btn_select_video.setEnabled(False)
        self.view.video_player.btn_force_stop.setEnabled(True)
        self.view.set_status_state("loading")

    def handle_upload(self) -> None:
        start_dir = str(Path(__file__).resolve().parents[3] / "videos")
        file_path, _ = QFileDialog.getOpenFileName(
            self.view,
            "选择比赛视频",
            start_dir,
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)",
        )
        if not file_path:
            self.view.append_log("[Info] 用户取消了视频选择。")
            return

        self._selected_video_path = file_path
        self.view.set_video_path(file_path)
        self.view.append_log(f"[Info] 正在读取视频信息: {Path(file_path).name}")

        self._probe_worker = ProbeWorker(file_path)
        self._probe_worker.finished.connect(self._on_probe_finished)
        self._probe_worker.start()

        self._start_tracknet_inference(file_path)

    def _on_probe_finished(self, file_path: str, metadata: dict) -> None:
        self._probe_worker = None
        if file_path != self._selected_video_path:
            return
        file_name = Path(file_path).name
        resolution = (
            f"{metadata['width']} x {metadata['height']}"
            if metadata["width"] and metadata["height"]
            else "分辨率未知"
        )
        self.view.set_video_state("loaded")
        self.view.append_log(
            f"[Info] 成功加载视频: {file_name} | {resolution} | FPS {metadata['fps']:.2f}"
        )

    def _start_tracknet_inference(self, video_path: str) -> None:
        if self._track_branch is None:
            self.view.append_log("[Error] TrackNet 模型未初始化")
            return

        if self._track_running:
            self.view.append_log("[Info] TrackNet 推理正在进行中...")
            self._track_running = False

        self._track_running = True
        self._set_running_state()
        self.view.append_log(f"[TrackNet] 开始推理: {Path(video_path).name}")
        self._run_tracknet_thread(video_path)

    def _run_tracknet_thread(self, video_path: str) -> None:
        def tracknet_loop():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.view.append_log(f"[Error] 无法打开视频: {video_path}")
                self._track_running = False
                return

            ok, first_frame = cap.read()
            if not ok:
                self.view.append_log("[Error] 无法读取视频帧")
                cap.release()
                self._track_running = False
                return

            ok, second_frame = cap.read()
            if not ok:
                second_frame = first_frame.copy()

            prev_frame = first_frame.copy()
            curr_frame = first_frame
            next_frame = second_frame
            tick_frequency = cv2.getTickCount()
            ema_fps = 0.0

            while self._track_running:
                if isdeleted(self.view):
                    break

                start_tick = cv2.getTickCount()
                _, track = self._track_branch.infer([prev_frame, curr_frame, next_frame])
                end_tick = cv2.getTickCount()
                elapsed = max((end_tick - start_tick) / tick_frequency, 1e-6)
                instant_fps = 1.0 / elapsed
                ema_fps = instant_fps if ema_fps == 0.0 else 0.9 * ema_fps + 0.1 * instant_fps

                if not isdeleted(self.view):
                    frame_copy = curr_frame.copy()
                    QTimer.singleShot(0, lambda f=frame_copy, t=track, fps=ema_fps:
                                   self.view.update_tracknet_frame(f, t, fps))

                prev_frame = curr_frame
                curr_frame = next_frame
                ok, incoming = cap.read()
                if not ok:
                    break
                next_frame = incoming

            cap.release()
            self._track_running = False
            if not isdeleted(self.view):
                QTimer.singleShot(0, lambda: (
                    self.view.append_log("[TrackNet] 推理完成"),
                    self._set_idle_state()
                ))

        thread = Thread(target=tracknet_loop, daemon=True)
        thread.start()

    def handle_force_stop(self) -> None:
        if self._track_running:
            self.view.append_log("[Info] 正在停止 TrackNet 推理...")
            self._track_running = False
            self._set_idle_state()
        else:
            self.view.append_log("[Info] 视频预览已暂停。")

        self.view.video_player.pause()

    def handle_reset(self) -> None:
        if self._probe_worker is not None:
            if self._probe_worker.isRunning():
                self._probe_worker.quit()
                self._probe_worker.wait(500)
            self._probe_worker = None

        if self._track_running:
            self._track_running = False
            self.view.append_log("[TrackNet] 推理已停止")

        self._selected_video_path = None
        self.view.clear_video()
        self.view.log_console.clear()
        self.view.append_log("[System] 工作区已重置。")
        self._set_idle_state()

    def _on_style_action_triggered(self, action) -> None:
        theme_name = str(action.data() or action.text()).strip()
        theme_label = action.text().strip() or theme_name.replace("_", " ").title()
        self.view.style_btn.setText(f"{theme_label}  ▾")
        self.handle_style_changed(theme_name)

    def handle_style_changed(self, theme_name: str) -> None:
        app = QApplication.instance()
        if app is None or not theme_name.strip():
            return

        theme_dir = next(
            (d for d in self._theme_dirs if d.name == theme_name),
            None,
        )
        if theme_dir is None:
            return

        def _apply() -> None:
            apply_theme(app, theme_dir)
            self.view.append_log(f"[Theme] 已切换主题: {theme_dir.name}")

        QTimer.singleShot(0, _apply)


MockController = MainController
