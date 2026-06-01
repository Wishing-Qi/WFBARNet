from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QPushButton
from PySide6.QtCore import Qt, QPoint, QRect, Signal, Slot
from PySide6.QtGui import QPainter, QPen, QColor, QImage, QPixmap, QFont, QBrush
from typing import List, Dict, Any, Optional, Tuple
from common.packet import FramePacket
import numpy as np

class VideoWidget(QLabel):
    """
    高性能视频渲染组件，支持在帧上直接绘制检测结果与手动标定。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        
        # 骨架连通图 (COCO 格式 17 点)
        self.skeleton_pairs = [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
        ]

        # 标定相关
        self.calibration_mode = False
        self.corners = [] # [(x, y), ...] 像素坐标
        self.dragging_idx = -1
        self.current_packet = None
        self.scale_ratio = 1.0
        self.target_rect = QRect()
        self._last_sent_corners = None

        # 渲染开关
        self.show_skeletons = True
        self.show_ball = True
        self.show_court = True
        self.show_labels = True

    def set_calibration_mode(self, enabled: bool):
        self.calibration_mode = enabled
        if enabled and not self.corners:
            # 如果进入标定模式时没有角点，通过当前帧尺寸初始化 4 个默认点
            if self.current_packet and self.current_packet.image is not None:
                h, w = self.current_packet.image.shape[:2]
            else:
                w, h = 1280, 720 # 默认兜底
            
            # 初始化为画面中心的矩形 (10% 边距)
            self.corners = [
                (w * 0.1, h * 0.1), (w * 0.9, h * 0.1),
                (w * 0.9, h * 0.9), (w * 0.1, h * 0.9)
            ]
            
        self.update()

    def draw_packet(self, packet: FramePacket):
        self.current_packet = packet
        
        # 自动初始化默认角点
        if not self.corners and packet.image is not None:
            h, w = packet.image.shape[:2]
            self.corners = [
                (w * 0.1, h * 0.1), (w * 0.9, h * 0.1),
                (w * 0.9, h * 0.9), (w * 0.1, h * 0.9)
            ]
        
        # 将当前的角点信息同步回 packet，供后端的 CourtMapper 使用
        if self.corners:
            packet.court_info = {"corners": self.corners}

            # 同步更新引擎侧映射，避免 UI 与检测帧不同步
            if self._last_sent_corners != self.corners:
                from core.engine.orchestrator import InferenceEngine
                InferenceEngine().court_mapper.update_homography(self.corners)
                self._last_sent_corners = list(self.corners)

        if not self.calibration_mode:
            # 如果不是标定模式，同步更新角点数据
            if packet.court_info and "corners" in packet.court_info:
                self.corners = packet.court_info["corners"]
        
        self.update()

    def set_frame(self, frame_packet: Optional[FramePacket]):
        """清空或重置当前显示的帧"""
        self.current_packet = frame_packet
        if frame_packet is None:
            self.corners = []
        self.update()

    def paintEvent(self, event):
        if not self.current_packet or self.current_packet.image is None:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 计算缩放后的图像绘制区域 (保持比例居中)
        frame = self.current_packet.image
        h, w, _ = frame.shape
        label_w, label_h = self.width(), self.height()
        
        scale = min(label_w / w, label_h / h)
        nw, nh = int(w * scale), int(h * scale)
        dx, dy = (label_w - nw) // 2, (label_h - nh) // 2
        
        self.target_rect = QRect(dx, dy, nw, nh)
        self.scale_ratio = scale

        # 渲染底层视频帧
        qt_img = QImage(frame.data, w, h, w * 3, QImage.Format_BGR888)
        painter.drawImage(self.target_rect, qt_img)

        # 2. 绘制检测结果 (映射到 Label 坐标系)
        self._draw_overlay(painter, dx, dy, scale)

        # 3. 绘制层 - 标定绿点
        self._draw_calibration_points(painter, dx, dy, scale)

        # 显式结束绘制，防止 QBackingStore 警告
        painter.end()

    def _draw_overlay(self, painter, dx, dy, scale):
        if not self.current_packet: return
        
        # 1. 绘制场线与轮廓
        if self.show_court and self.corners and len(self.corners) == 4:
            painter.setPen(QPen(QColor(0, 255, 0, 180), 2, Qt.SolidLine))
            pts = [QPoint(int(c[0] * scale + dx), int(c[1] * scale + dy)) for c in self.corners]
            painter.drawPolygon(pts)

        # 2. 绘制骨架与球员信息
        if self.show_skeletons:
            pen_joint = QPen(QColor(0, 255, 255), 4)
            pen_line = QPen(QColor(255, 255, 0), 2)
            pen_bbox = QPen(QColor(0, 255, 0, 120), 2)
            
            for skel in self.current_packet.skeletons:
                kpts = np.array(skel.get("keypoints", [])).reshape(-1, 3)
                bbox = skel.get("bbox")
                
                # 绘制 Bounding Box (作为最基础的可视化反馈)
                if bbox is not None and len(bbox) >= 4:
                    painter.setPen(pen_bbox)
                    x1, y1, x2, y2 = bbox[:4]
                    painter.drawRect(QRect(
                        int(x1 * scale + dx), int(y1 * scale + dy),
                        int((x2 - x1) * scale), int((y2 - y1) * scale)
                    ))
                
                # 画连线
                painter.setPen(pen_line)
                for p1, p2 in self.skeleton_pairs:
                    if kpts[p1, 2] > 0.3 and kpts[p2, 2] > 0.3:
                        painter.drawLine(
                            int(kpts[p1, 0] * scale + dx), int(kpts[p1, 1] * scale + dy),
                            int(kpts[p2, 0] * scale + dx), int(kpts[p2, 1] * scale + dy)
                        )
                
                # 画关键点
                painter.setPen(pen_joint)
                for x, y, conf in kpts:
                    if conf > 0.3:
                        painter.drawPoint(int(x * scale + dx), int(y * scale + dy))

                # 画标签 (ID & 动作)
                if self.show_labels:
                    tid = skel.get("player_id", "N/A")
                    action = skel.get("action", "Normal")
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(QFont("Arial", 10, QFont.Bold))
                    painter.drawText(int(kpts[0, 0] * scale + dx), int(kpts[0, 1] * scale + dy - 20), 
                                   f"ID:{tid} [{action}]")

        # 3. 绘制球及拖尾
        if self.show_ball:
            ball_history = self.current_packet.metadata.get('ball_history', [])
            if ball_history:
                # 过滤掉 None 值，仅保留坐标
                valid_path = [(i, c) for i, c in enumerate(ball_history) if c is not None]
                num_total = len(ball_history)
                
                # 绘制连接线 (更加顺滑)
                if len(valid_path) > 1:
                    for idx in range(len(valid_path) - 1):
                        orig_idx1, coord1 = valid_path[idx]
                        orig_idx2, coord2 = valid_path[idx + 1]
                        
                        # 只有当两个点索引很近时才连线，防止丢帧严重的跳跃式连线
                        if (orig_idx2 - orig_idx1) < 5: 
                            # 即使是最老的点也保持至少 30 的透明度
                            alpha = int(30 + 190 * (orig_idx2 / num_total))
                            painter.setPen(QPen(QColor(255, 255, 0, alpha), 2, Qt.SolidLine))
                            painter.drawLine(
                                int(coord1[0] * scale + dx), int(coord1[1] * scale + dy),
                                int(coord2[0] * scale + dx), int(coord2[1] * scale + dy)
                            )

                # 绘制节点
                for i, coord in enumerate(ball_history):
                    if coord is None: continue
                    
                    px, py = int(coord[0] * scale + dx), int(coord[1] * scale + dy)
                    if i == num_total - 1:
                        # 当前球：亮红色核心
                        painter.setBrush(QColor(255, 0, 0))
                        painter.setPen(QPen(QColor(255, 255, 255), 2))
                        painter.drawEllipse(QPoint(px, py), 7, 7)
                    else:
                        # 拖尾点：黄色渐变，最小透明度 20
                        alpha = int(20 + 180 * (i + 1) / num_total)
                        size = max(1, int(5 * (i + 1) / num_total))
                        painter.setBrush(QColor(255, 255, 0, alpha))
                        painter.setPen(Qt.NoPen)
                        painter.drawEllipse(QPoint(px, py), size, size)
            
            elif self.current_packet.ball_coord:
                # 兜底：单点
                bx, by = self.current_packet.ball_coord[0], self.current_packet.ball_coord[1]
                painter.setBrush(QColor(255, 0, 0))
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.drawEllipse(QPoint(int(bx * scale + dx), int(by * scale + dy)), 7, 7)
            
        # 4. 状态信息绘制 (不受开关控制)
        painter.setPen(QColor(0, 255, 0))
        painter.setFont(QFont("Consolas", 10))
        fps = self.current_packet.metadata.get('infer_fps', 0.0)
        painter.drawText(20, 30, f"AI 推理: {fps:.1f} FPS")
        
        if self.calibration_mode:
            painter.setPen(QColor(255, 165, 0))
            painter.setFont(QFont("Arial", 14, QFont.Bold))
            painter.drawText(20, 60, "模式: 手动标定 (拖拽角点)")

    def _draw_calibration_points(self, painter, dx, dy, scale):
        if not self.corners: return
        
        pen_corner = QPen(QColor(0, 255, 0), 2)
        brush_corner = QBrush(QColor(0, 255, 0, 150))
        painter.setPen(pen_corner)
        
        for i, (cx, cy) in enumerate(self.corners):
            px, py = int(cx * scale + dx), int(cy * scale + dy)
            painter.setBrush(brush_corner if i != self.dragging_idx else QColor(255, 255, 255))
            painter.drawEllipse(QPoint(px, py), 8, 8)
            
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(px + 10, py + 10, str(i+1))
            painter.setPen(pen_corner)

    def mousePressEvent(self, event):
        if not self.calibration_mode or not self.corners: 
            super().mousePressEvent(event)
            return
        
        m_pos = event.pos()
        for i, (cx, cy) in enumerate(self.corners):
            px, py = int(cx * self.scale_ratio + self.target_rect.x()), int(cy * self.scale_ratio + self.target_rect.y())
            if (m_pos.x() - px)**2 + (m_pos.y() - py)**2 < 15**2:
                self.dragging_idx = i
                break

    def mouseMoveEvent(self, event):
        if self.dragging_idx != -1:
            m_pos = event.pos()
            orig_x = (m_pos.x() - self.target_rect.x()) / self.scale_ratio
            orig_y = (m_pos.y() - self.target_rect.y()) / self.scale_ratio
            self.corners[self.dragging_idx] = (orig_x, orig_y)
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragging_idx = -1
        super().mouseReleaseEvent(event)

class VideoDisplayContainer(QWidget):
    """
    组合组件：包含 VideoWidget 和底部的控制栏/播放进度条。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_paused = False # 播放状态缓存

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        self.display = VideoWidget()
        layout.addWidget(self.display, stretch=1)

        ctrl_widget = QWidget()
        ctrl_widget.setFixedHeight(45)
        ctrl_layout = QHBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(10, 0, 10, 5)
        
        self.btn_play = QPushButton("⏸") # 默认为播放状态
        self.btn_play.setFixedSize(32, 28)
        self.btn_play.clicked.connect(self.toggle_play_pause)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setStyleSheet("""
            QSlider::handle:horizontal { width: 12px; height: 12px; margin: -5px 0; }
            QSlider::groove:horizontal { height: 4px; background: #DDD; }
        """)
        # 绑定进度条拖拽信号 (使用 sliderReleased 减轻引擎负担)
        self.slider.sliderReleased.connect(self.on_slider_seek)
        
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font-size: 11px;")
        
        self.btn_calibrate = QPushButton("手动标定")
        self.btn_calibrate.setFixedSize(80, 28)
        self.btn_calibrate.setCheckable(True)
        self.btn_calibrate.toggled.connect(self.display.set_calibration_mode)

        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.slider)
        ctrl_layout.addWidget(self.lbl_time)
        ctrl_layout.addWidget(self.btn_calibrate)
        
        layout.addWidget(ctrl_widget)

    def toggle_play_pause(self):
        self.is_paused = not self.is_paused
        self.btn_play.setText("▶" if self.is_paused else "⏸")
        # 通过 InferenceEngine 单例控制后端
        from core.engine.orchestrator import InferenceEngine
        engine = InferenceEngine()
        engine.paused = self.is_paused

    def on_slider_seek(self):
        val = self.slider.value() / 1000.0
        from core.engine.orchestrator import InferenceEngine
        engine = InferenceEngine()
        if engine.has_source() and hasattr(engine.stream_plugin, 'cap'):
            total = engine.stream_plugin.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            engine.stream_plugin.cap.set(cv2.CAP_PROP_POS_FRAMES, int(val * total))

    def update_frame(self, packet: FramePacket):
        self.display.draw_packet(packet)
