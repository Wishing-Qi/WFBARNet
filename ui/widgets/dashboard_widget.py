from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QLabel, QTableWidget, QTableWidgetItem, QHeaderView, 
                             QScrollArea, QFrame)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QColor

class KPICard(QFrame):
    """
    数据指标卡片组件
    """
    def __init__(self, title, value, unit="", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        # 允许样式表控制自定义挂件
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        layout = QVBoxLayout(self)
        self.lbl_title = QLabel(title)
        self.lbl_title.setObjectName("Title")
        
        value_layout = QHBoxLayout()
        self.lbl_value = QLabel(str(value))
        self.lbl_value.setObjectName("Value")
        self.lbl_unit = QLabel(unit)
        self.lbl_unit.setObjectName("Unit")
        value_layout.addWidget(self.lbl_value)
        value_layout.addWidget(self.lbl_unit)
        value_layout.addStretch()
        
        layout.addWidget(self.lbl_title)
        layout.addLayout(value_layout)

    def set_value(self, value):
        self.lbl_value.setText(str(value))

class DashboardWidget(QWidget):
    """
    右侧概览仪表盘组件
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        
        # 1. 顶部 2x2 KPI 卡片区
        self.kpi_container = QWidget()
        self.kpi_grid = QGridLayout(self.kpi_container)
        self.kpi_grid.setContentsMargins(0, 0, 0, 0)
        
        self.card_fps = KPICard("实时监控 FPS", "0.0", "FPS")
        self.card_infer = KPICard("AI 推理 FPS", "0.0", "FPS")
        self.card_status = KPICard("回合状态", "准备中")
        self.card_hits = KPICard("击球总数", "0", "次")
        
        self.kpi_grid.addWidget(self.card_fps, 0, 0)
        self.kpi_grid.addWidget(self.card_infer, 0, 1)
        self.kpi_grid.addWidget(self.card_status, 1, 0)
        self.kpi_grid.addWidget(self.card_hits, 1, 1)
        
        self.main_layout.addWidget(self.kpi_container)
        
        # 2. 动作时序列表区
        self.lbl_table_dir = QLabel("动作时序识别结果")
        self.lbl_table_dir.setObjectName("ActionTableTitle")
        self.lbl_table_dir.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.main_layout.addWidget(self.lbl_table_dir)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["时间段", "动作类别", "置信度", "动作细节"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        
        self.main_layout.addWidget(self.table)

    @Slot(dict)
    def update_metrics(self, data: dict):
        """
        更新 KPI 指标数据
        :param data: {'fps': 30, 'infer_fps': 25, 'rally': True, 'hits': 10}
        """
        if 'fps' in data: self.card_fps.set_value(f"{data['fps']:.1f}")
        if 'infer_fps' in data: self.card_infer.set_value(f"{data['infer_fps']:.1f}")
        if 'rally' in data: 
            status = "比赛进行中" if data['rally'] else "回合停止"
            self.card_status.set_value(status)
        if 'hits' in data: self.card_hits.set_value(data['hits'])

    @Slot(dict)
    def add_action_row(self, action_info: dict, insert_at_top: bool = False):
        """
        向表格添加一行动作识别结果
        :param action_info: {'time': '00:12-00:14', 'type': '扣杀', 'conf': 0.98, 'detail': 'Player 1'}
        :param insert_at_top: 是否在此表格最上方插入
        """
        self.table.setUpdatesEnabled(False) # 性能优化：禁用界面重绘
        try:
            row_idx = 0 if insert_at_top else self.table.rowCount()
            self.table.insertRow(row_idx)
            
            # 创建条目
            items = [
                QTableWidgetItem(action_info.get('time', '-')),
                QTableWidgetItem(action_info.get('type', '未知')),
                QTableWidgetItem(f"{action_info.get('conf', 0.0)*100:.1f}%"),
                QTableWidgetItem(action_info.get('detail', '-'))
            ]
            
            # 针对特定动作进行颜色区分 (例如：扣杀使用红色强调)
            if action_info.get('type') == '扣杀':
                for item in items:
                    item.setBackground(QColor(183, 28, 28, 100)) # 深红色背景
                    item.setForeground(QColor(255, 255, 255))
            
            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_idx, col, item)
                
            # 如果是追加到末尾则滚动到底部，如果是插入到顶部则确保首行可见
            if insert_at_top:
                self.table.scrollToTop()
            else:
                self.table.scrollToBottom()
        finally:
            self.table.setUpdatesEnabled(True) # 恢复界面重绘
