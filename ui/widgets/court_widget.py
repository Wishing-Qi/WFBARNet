from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from common.packet import FramePacket

class CourtWidget(QWidget):
    """
    2D 战术看板：展示羽毛球场俯视图及球员/球的实时物理位置。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 600)
        self.court_w_meters = 6.1
        self.court_l_meters = 13.4
        self.margin = 20

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect().adjusted(self.margin, self.margin, -self.margin, -self.margin)
        
        # 1. 绘制球场底色
        painter.setBrush(QColor(34, 139, 34)) # 森林绿
        painter.drawRect(rect)
        
        # 2. 绘制球场线 (白线)
        painter.setPen(QPen(Qt.white, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect) # 外框
        
        # 中间网线
        mid_y = rect.top() + rect.height() / 2
        painter.drawLine(rect.left(), mid_y, rect.right(), mid_y)
        
        # 简单的发球线逻辑 (演示用)
        # TODO: 根据 court_cfg 绘制更精确的线条

    def update_positions(self, packet: FramePacket, mapper):
        """
        根据物理坐标更新点位。
        """
        # 注意：此处通常需要配合重写 paintEvent 来绘制动态点，
        # 或者在 update_positions 中存储数据并触发 repaint()。
        self.current_packet = packet
        self.mapper = mapper
        self.update() # 触发重绘
