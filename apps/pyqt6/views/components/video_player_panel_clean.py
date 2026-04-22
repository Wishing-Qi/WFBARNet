from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class VideoPlayerWidget(QFrame):
    """视频预览区，仅包含标题、状态和画面。"""

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

        placeholder_page = QWidget()
        placeholder_layout = QVBoxLayout(placeholder_page)
        placeholder_layout.setContentsMargins(0, 0, 0, 0)

        placeholder_frame = QFrame()
        placeholder_frame.setObjectName("videoPlaceholderFrame")
        placeholder_frame_layout = QVBoxLayout(placeholder_frame)
        placeholder_frame_layout.setContentsMargins(0, 0, 0, 0)
        placeholder_frame_layout.setSpacing(0)

        placeholder_layout.addWidget(placeholder_frame)

        self.video_widget = QVideoWidget()
        self.video_widget.setObjectName("videoOutput")
        self.preview_stack.addWidget(placeholder_page)
        self.preview_stack.addWidget(self.video_widget)
        self.preview_stack.setCurrentWidget(placeholder_page)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.0)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)

        outer.addWidget(self.preview_stack, stretch=1)

        self.btn_select_video.clicked.connect(self.selectRequested.emit)
        self.btn_force_stop.clicked.connect(self.forceStopRequested.emit)

        self.player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)

    def _set_status(self, text: str, state: str) -> None:
        pass  # state_label 已移除

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.LoadedMedia and self._source_path:
            self.preview_stack.setCurrentWidget(self.video_widget)

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        pass  # 可在此处添加播放状态变化时的处理逻辑

    def set_video_path(self, path: str) -> None:
        self._source_path = path
        self.path_edit.setText(path)
        self.path_edit.setToolTip(path)
        self.player.setSource(QUrl.fromLocalFile(path))

    def clear_video(self) -> None:
        self._source_path = ""
        self.path_edit.clear()
        self.path_edit.setToolTip("")
        self.player.stop()
        self.player.setSource(QUrl())
        self.preview_stack.setCurrentWidget(self.preview_stack.widget(0))
        self._set_status("未加载视频", "idle")

    def play(self) -> None:
        if self._source_path:
            self.preview_stack.setCurrentWidget(self.video_widget)
            self.player.play()

    def pause(self) -> None:
        self.player.pause()

    def stop(self) -> None:
        self.player.stop()
        if self._source_path:
            self._set_status("已停止", "stopped")
        else:
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setObjectName("videoTimeline")
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setSingleStep(1000)
        self.seek_slider.setPageStep(5000)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("timeLabel")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.seek_slider, stretch=1)
        layout.addWidget(self.time_label)

        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.positionChanged.connect(self._on_position_changed)

    def _format_time(self, milliseconds: int) -> str:
        total_seconds = max(0, milliseconds // 1000)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        return f"{minutes:02}:{seconds:02}"

    def _on_slider_pressed(self) -> None:
        self._dragging = True

    def _on_slider_released(self) -> None:
        self._dragging = False
        self._player.setPosition(self.seek_slider.value())

    def _on_slider_moved(self, value: int) -> None:
        self._player.setPosition(value)
        self.time_label.setText(
            f"{self._format_time(value)} / {self._format_time(self._duration_ms)}"
        )

    def _on_duration_changed(self, duration: int) -> None:
        self._duration_ms = max(0, duration)
        self.seek_slider.setRange(0, self._duration_ms)
        self.time_label.setText(
            f"{self._format_time(self._player.position())} / {self._format_time(self._duration_ms)}"
        )

    def _on_position_changed(self, position: int) -> None:
        if not self._dragging:
            self.seek_slider.setValue(position)
        self.time_label.setText(
            f"{self._format_time(position)} / {self._format_time(self._duration_ms)}"
        )

    def reset(self) -> None:
        self._dragging = False
        self._duration_ms = 0
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")
