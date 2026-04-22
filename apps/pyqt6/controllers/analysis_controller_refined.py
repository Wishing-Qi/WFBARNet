from __future__ import annotations

from pathlib import Path

import cv2
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from apps.pyqt6.models.analysis_service import AnalysisService
from apps.pyqt6.models.analysis_types import AnalysisResult
from apps.pyqt6.utils.style import apply_theme, discover_themes
from apps.pyqt6.views.main_window_refined import MainWindow


class AnalysisWorker(QThread):
    progress_payload = pyqtSignal(object)
    finished_with_result = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, service: AnalysisService, video_path: str) -> None:
        super().__init__()
        self._service = service
        self._video_path = video_path
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            result = self._service.analyze_video(
                self._video_path,
                progress_callback=lambda payload: self.progress_payload.emit(payload),
                stop_requested=lambda: self._stop_requested,
            )
            if self._stop_requested:
                result.status = "stopped"
                result.message = "分析已中止"
            self.finished_with_result.emit(result)
        except Exception as exc:  # pragma: no cover - UI error path
            self.failed.emit(str(exc))


class MainController:
    """控制层：连接视频选择、强制停止、分析线程和结果回填。"""

    def __init__(self, view: MainWindow) -> None:
        self.view = view
        self.service = AnalysisService()
        self._selected_video_path: str | None = None
        self._worker: AnalysisWorker | None = None
        self._theme_dirs = discover_themes()

        self._bind_events()
        self.view.populate_stylesheets(self._theme_dirs)
        self._set_idle_state()
        self.view.append_log("[System] 界面已初始化，等待用户操作。")

    def _bind_events(self) -> None:
        self.view.btn_analyze.clicked.connect(self.handle_analyze)
        self.view.btn_reset.clicked.connect(self.handle_reset)
        self.view.video_player.selectRequested.connect(self.handle_upload)
        self.view.video_player.forceStopRequested.connect(self.handle_force_stop)
        self.view._style_menu.triggered.connect(self._on_style_action_triggered)

    def _set_idle_state(self) -> None:
        self.view.btn_analyze.setEnabled(self._selected_video_path is not None)
        self.view.btn_reset.setEnabled(True)
        self.view.video_player.btn_select_video.setEnabled(True)
        self.view.video_player.btn_force_stop.setEnabled(True)
        self.view.set_status_state("idle")

    def _set_running_state(self) -> None:
        self.view.btn_analyze.setEnabled(False)
        self.view.btn_reset.setEnabled(True)
        self.view.video_player.btn_select_video.setEnabled(False)
        self.view.video_player.btn_force_stop.setEnabled(True)
        self.view.set_status_state("loading")

    def _set_success_state(self) -> None:
        self.view.btn_analyze.setEnabled(self._selected_video_path is not None)
        self.view.btn_reset.setEnabled(True)
        self.view.video_player.btn_select_video.setEnabled(True)
        self.view.video_player.btn_force_stop.setEnabled(True)
        self.view.set_status_state("success")

    def _set_stopped_state(self) -> None:
        self.view.btn_analyze.setEnabled(self._selected_video_path is not None)
        self.view.btn_reset.setEnabled(True)
        self.view.video_player.btn_select_video.setEnabled(True)
        self.view.video_player.btn_force_stop.setEnabled(True)
        self.view.set_status_state("stopped")

    def handle_upload(self) -> None:
        start_dir = str(self.service.project_root / "videos")
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
        metadata = self._probe_video_metadata(file_path)
        file_name = Path(file_path).name
        resolution = (
            f"{metadata['width']} x {metadata['height']}"
            if metadata["width"] and metadata["height"]
            else "分辨率未知"
        )

        self.view.set_video_path(file_path)
        self.view.set_video_state("loaded")
        self.view.btn_analyze.setEnabled(True)
        self.view.append_log(
            f"[Info] 成功加载本地视频: {file_name} | {resolution} | FPS {metadata['fps']:.2f}"
        )

    def handle_analyze(self) -> None:
        if not self._selected_video_path:
            default_path = self.service.project_root / "videos" / "sample.mp4"
            if default_path.exists():
                self._selected_video_path = str(default_path)
                self.view.set_video_path(str(default_path))
                self.view.set_video_state("loaded")
                self.view.append_log("[Info] 未选择视频，已自动使用示例视频 sample.mp4。")
            else:
                self.view.append_log("[Warn] 请先选择一个视频文件，再开始分析。")
                QMessageBox.warning(self.view, "未选择视频", "请先选择一个视频文件，再开始分析。")
                return

        if self._worker is not None and self._worker.isRunning():
            self.view.append_log("[Warn] 当前已有分析任务在运行。")
            return

        self._set_running_state()
        self.view.video_player.pause()
        self.view.tabs.setCurrentIndex(0)
        self.view.append_log("[System] 分析任务已启动。")

        self._worker = AnalysisWorker(self.service, self._selected_video_path)
        self._worker.progress_payload.connect(self._handle_progress_payload)
        self._worker.finished_with_result.connect(self._handle_analysis_finished)
        self._worker.failed.connect(self._handle_analysis_failed)
        self._worker.start()

    def handle_force_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.view.append_log("[Warn] 正在强制停止分析任务...")
            self._worker.request_stop()
            if not self._worker.wait(400):
                self._worker.terminate()
                self._worker.wait(1500)
            self._worker = None
            self.view.append_log("[System] 分析任务已强制停止。")
        else:
            self.view.append_log("[System] 视频预览已暂停。")

        self.view.video_player.pause()
        self._set_stopped_state()

    def _handle_progress_payload(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        stage = str(payload.get("stage", ""))
        progress = float(payload.get("progress", 0.0))
        self.view.update_progress(max(0, min(int(progress * 100), 100)))
        if stage:
            self.view.status_label.setText(f"系统状态：{stage}")
            self.view.append_log(f"[Progress] {stage}")

    def _handle_analysis_finished(self, result: object) -> None:
        analysis_result = AnalysisResult.from_payload(result)
        self._worker = None

        if analysis_result.status == "success":
            self._render_analysis_result(analysis_result)
            self.view.update_progress(100)
            self._set_success_state()
            self.view.status_label.setText("系统状态：分析完成")
            self.view.append_log(f"[Success] {analysis_result.message}")
            return

        self._set_stopped_state()
        self.view.status_label.setText("系统状态：已中止")
        self.view.append_log(f"[Warn] {analysis_result.message}")

    def _handle_analysis_failed(self, error_message: str) -> None:
        self._worker = None
        self._set_stopped_state()
        self.view.set_status_state("error")
        self.view.status_label.setText("系统状态：分析失败")
        self.view.append_log(f"[Error] {error_message}")
        QMessageBox.critical(self.view, "分析失败", error_message)

    def _render_analysis_result(self, result: AnalysisResult) -> None:
        self.view.tabs.setCurrentIndex(0)
        self.view.lbl_total_actions.setText(str(result.action_count))
        self.view.lbl_avg_conf.setText(f"{result.avg_confidence * 100:.1f}%")
        self.view.lbl_valid_pose.setText(str(result.valid_pose_frames))
        self.view.lbl_valid_track.setText(str(result.valid_track_frames))

        self.view.table_actions.setRowCount(0)
        for action in result.actions:
            self.view.add_action_row(action.time_range, action.label, action.confidence, action.detail)

        self.view.append_log(f"[Info] 输出目录: {result.output_dir}")
        for name, path in result.output_files.items():
            self.view.append_log(f"[File] {name}: {path}")
        self.view.append_log(result.to_display_text())

    def handle_reset(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.view.append_log("[Warn] 重置前先终止当前任务...")
            self._worker.request_stop()
            if not self._worker.wait(400):
                self._worker.terminate()
                self._worker.wait(1500)
            self._worker = None

        self._selected_video_path = None
        self.view.clear_video()
        self.view.video_timeline.reset()
        self.view.reset_analysis()
        self.view.log_console.clear()
        self.view.append_log("[System] 工作区已重置。")
        self._set_idle_state()

    def _on_style_action_triggered(self, action) -> None:
        theme_name = action.data()
        self.view.style_btn.setText(f"{theme_name}  ▾")
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

    def _probe_video_metadata(self, file_path: str) -> dict[str, float | int]:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return {"fps": 0.0, "width": 0, "height": 0}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return {"fps": fps, "width": width, "height": height}


MockController = MainController
