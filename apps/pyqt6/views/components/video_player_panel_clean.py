from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoFrame, QVideoSink
from PyQt6.sip import isdeleted
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class VideoPlayerWidget(QFrame):
    """视频预览区，使用 QVideoSink + QLabel 手动渲染，避免 WMF 黑屏。"""

    selectRequested = pyqtSignal()
    forceStopRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoPlayerCard")

        self._dragging = False
        self._source_path = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.btn_select_video = QPushButton("选择视频")
        self.btn_select_video.setObjectName("btnSelectVideo")

        self.path_edit = QLineEdit()
        self.path_edit.setObjectName("videoPathEdit")
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("视频路径")
        self.path_edit.setClearButtonEnabled(False)

        self.btn_force_stop = QPushButton("强制停止")
        self.btn_force_stop.setObjectName("btnForceStop")

        self.preview_stack = QStackedWidget()
        self.preview_stack.setObjectName("videoStack")
        self.preview_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # 占位页
        placeholder_page = QWidget()
        placeholder_layout = QVBoxLayout(placeholder_page)
        placeholder_layout.setContentsMargins(0, 0, 0, 0)

        placeholder_frame = QFrame()
        placeholder_frame.setObjectName("videoPlaceholderFrame")
        placeholder_frame_layout = QVBoxLayout(placeholder_frame)
        placeholder_frame_layout.setContentsMargins(0, 0, 0, 0)
        placeholder_frame_layout.setSpacing(0)

        placeholder_layout.addWidget(placeholder_frame)

        # 用 QLabel 替代 QVideoWidget，手动渲染视频帧
        self.video_label = QLabel()
        self.video_label.setObjectName("videoOutput")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.video_label.setStyleSheet("background-color: black;")

        self.preview_stack.addWidget(placeholder_page)
        self.preview_stack.addWidget(self.video_label)
        self.preview_stack.setCurrentWidget(placeholder_page)

        # QMediaPlayer + QVideoSink（不再使用 QVideoWidget）
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.0)
        self.player.setAudioOutput(self.audio_output)

        self._video_sink = QVideoSink(self)
        self.player.setVideoOutput(self._video_sink)
        self._video_sink.videoFrameChanged.connect(self._on_video_frame)

        outer.addWidget(self.preview_stack, stretch=1)

        self.btn_select_video.clicked.connect(self.selectRequested.emit)
        self.btn_force_stop.clicked.connect(self.forceStopRequested.emit)

        self.player.mediaStatusChanged.connect(self._on_media_status_changed)

    def _on_video_frame(self, frame: QVideoFrame) -> None:
        """每收到一帧就转为 QPixmap 绘制到 QLabel 上。"""
        if frame.isValid() and not isdeleted(self.video_label):
            image = frame.toImage()
            if not image.isNull():
                pixmap = QPixmap.fromImage(image)
                scaled = pixmap.scaled(
                    self.video_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.video_label.setPixmap(scaled)

    def _set_status(self, text: str, state: str) -> None:
        pass  # state_label 已移除

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.LoadedMedia and self._source_path:
            self.preview_stack.setCurrentWidget(self.video_label)
            # play 再 pause 触发第一帧渲染
            self.player.play()
            self.player.pause()

    def set_video_path(self, path: str) -> None:
        self._source_path = path
        self.path_edit.setText(path)
        self.path_edit.setToolTip(path)
        self.player.setSource(QUrl.fromLocalFile(path))

    def clear_video(self) -> None:
        self._source_path = ""
        if not isdeleted(self.path_edit):
            self.path_edit.clear()
            self.path_edit.setToolTip("")
        self.player.stop()
        self.player.setSource(QUrl())
        if not isdeleted(self.video_label):
            self.video_label.clear()
        if not isdeleted(self.preview_stack) and self.preview_stack.count() > 0:
            self.preview_stack.setCurrentWidget(self.preview_stack.widget(0))
        self._set_status("未加载视频", "idle")

    def play(self) -> None:
        if self._source_path and not isdeleted(self.preview_stack):
            self.preview_stack.setCurrentWidget(self.video_label)
            self.player.play()

    def pause(self) -> None:
        self.player.pause()

    def stop(self) -> None:
        self.player.stop()
        if self._source_path:
            self._set_status("已停止", "stopped")
        elif not isdeleted(self.preview_stack) and self.preview_stack.count() > 0:
            self.preview_stack.setCurrentWidget(self.preview_stack.widget(0))
            self._set_status("未加载视频", "idle")

    def set_video_state(self, state: str) -> None:
        mapping = {
            "idle": "未加载视频",
            "loaded": "已加载视频",
            "playing": "播放中",
            "stopped": "已停止",
            "error": "视频加载失败",
        }
        self._set_status(mapping.get(state, state), state)

    def current_path(self) -> str:
        return self._source_path


class VideoTimelineWidget(QFrame):
    """视频时间轴，放在播放器外侧但仍属于视频预览卡片。"""

    def __init__(self, player: QMediaPlayer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoTimelineCard")
        self._player = player
        self._dragging = False
        self._duration_ms = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setObjectName("videoSeekSlider")
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("videoTimeLabel")
        self.time_label.setFixedWidth(110)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.seek_slider)
        layout.addWidget(self.time_label)

        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)

        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)

    def bind_player(self, player: QMediaPlayer) -> None:
        self._player = player
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)

    @staticmethod
    def _format_time(ms: int) -> str:
        total_sec = max(0, ms) // 1000
        return f"{total_sec // 60:02d}:{total_sec % 60:02d}"

    def _on_slider_pressed(self) -> None:
        self._dragging = True

    def _on_slider_released(self) -> None:
        self._dragging = False
        self._player.setPosition(self.seek_slider.value())

    def _on_slider_moved(self, value: int) -> None:
        self._player.setPosition(value)
        if not isdeleted(self.time_label):
            self.time_label.setText(
                f"{self._format_time(value)} / {self._format_time(self._duration_ms)}"
            )

    def _on_duration_changed(self, duration: int) -> None:
        self._duration_ms = max(0, duration)
        if not isdeleted(self.seek_slider):
            self.seek_slider.setRange(0, self._duration_ms)
        if not isdeleted(self.time_label):
            self.time_label.setText(
                f"{self._format_time(self._player.position())} / {self._format_time(self._duration_ms)}"
            )

    def _on_position_changed(self, position: int) -> None:
        if not self._dragging:
            if not isdeleted(self.seek_slider):
                self.seek_slider.setValue(position)
        if not isdeleted(self.time_label):
            self.time_label.setText(
                f"{self._format_time(position)} / {self._format_time(self._duration_ms)}"
            )

    def reset(self) -> None:
        self._dragging = False
        self._duration_ms = 0
        if not isdeleted(self.seek_slider):
            self.seek_slider.setRange(0, 0)
            self.seek_slider.setValue(0)
        if not isdeleted(self.time_label):
            self.time_label.setText("00:00 / 00:00")
