import sys
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QComboBox, 
                             QTabWidget, QSplitter, QStatusBar, QFrame, QSizePolicy,
                             QFileDialog, QLineEdit, QProgressDialog, QMessageBox)
from PySide6.QtCore import Qt, QSize, Slot
from PySide6.QtGui import QFont, QColor
from enum import Enum

from common.packet import FramePacket
from ui.thread_manager import EngineThread
from utils.logger import logger

class AppStatus(Enum):
    IDLE = ("待机中", "#757575")      # 灰色
    READY = ("就绪", "#1976D2")      # 蓝色
    ANALYZING = ("分析中", "#388E3C") # 绿色
    ERROR = ("异常", "#D32F2F")      # 红色

class ThemeManager:
    """主题管理类，支持 QSS 动态切换"""
    DARK_THEME = """
        QMainWindow { background-color: #1E1E1E; }
        QWidget { background-color: #1E1E1E; color: #E0E0E0; }
        QLabel { background: transparent; }
        QPushButton { background-color: #1B5E20; color: white; border-radius: 4px; padding: 8px; border: none; }
        QPushButton:hover { background-color: #2E7D32; }
        QTabWidget::pane { border: 1px solid #333; }
        QTabBar::tab { background: #252526; color: #AAA; padding: 10px; min-width: 80px; }
        QTabBar::tab:selected { background: #1B5E20; color: white; }
        QFrame#Header { background-color: #252526; border-bottom: 1px solid #333; }
        QTableWidget { background-color: #252526; color: #E0E0E0; gridline-color: #333; }
        QHeaderView::section { background-color: #333; color: #E0E0E0; border: 1px solid #444; }
        QLabel#MainTitle { color: #81C784; }  /* 在深色背景下使用浅绿色 */
        QLabel#StatusTag { color: #AAAAAA; }
        QLabel#ActionTableTitle { color: #81C784; }
        QComboBox { background-color: #333; color: #E0E0E0; border: 1px solid #444; padding: 4px; }
        QComboBox QAbstractItemView { background-color: #333; color: #E0E0E0; selection-background-color: #1B5E20; }
        
        KPICard { background-color: #2D2D2D; border: 1px solid #444; }
        KPICard QLabel#Title { color: #AAAAAA; }
        KPICard QLabel#Value { color: #00E676; }
        KPICard QLabel#Unit { color: #666666; }
    """
    
    LIGHT_THEME = """
        QMainWindow { background-color: #F5F5F5; }
        QWidget { background-color: #F5F5F5; color: #333333; }
        QLabel { background: transparent; }
        QPushButton { background-color: #1B5E20; color: white; border-radius: 4px; padding: 8px; font-weight: bold; border: none; }
        QPushButton:hover { background-color: #2E7D32; }
        QTabWidget::pane { border: 1px solid #DDDDDD; background: white; }
        QTabBar::tab { background: #E0E0E0; color: #333; padding: 10px; min-width: 80px; }
        QTabBar::tab:selected { background: white; border-bottom: 2px solid #1B5E20; }
        QFrame#Header { background-color: white; border-bottom: 1px solid #DDDDDD; }
        QTableWidget { background-color: white; color: #333; gridline-color: #DDD; }
        QHeaderView::section { background-color: #F0F0F0; color: #333; border: 1px solid #DDD; }
        QLabel#MainTitle { color: #1B5E20; }
        QLabel#StatusTag { color: #666666; }
        QLabel#ActionTableTitle { color: #1B5E20; }
        QComboBox { background-color: white; color: #333; border: 1px solid #DDD; padding: 4px; }

        KPICard { background-color: #FFFFFF; border: 1px solid #DDDDDD; }
        KPICard QLabel#Title { color: #666666; }
        KPICard QLabel#Value { color: #1B5E20; }
        KPICard QLabel#Unit { color: #999999; }
    """

class BadmintonMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WFBARNet - 羽毛球自动分析平台")
        self.resize(1440, 900)
        
        self.engine_thread = None
        
        self.setup_ui()
        self.apply_theme("Light")

    def setup_ui(self):
        # 主中央挂件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 1. 顶部状态区 (Header)
        self.header_frame = QFrame()
        self.header_frame.setObjectName("Header")
        self.header_frame.setFixedHeight(80)
        header_layout = QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(20, 0, 20, 0)

        # 标题与副标题
        title_container = QVBoxLayout()
        self.lbl_title = QLabel("羽毛球动作分析平台")
        self.lbl_title.setObjectName("MainTitle")
        self.lbl_title.setStyleSheet("font-size: 20px; font-weight: bold;")
        
        status_layout = QHBoxLayout()
        self.lbl_status_dot = QLabel("●")
        self.lbl_status_dot.setObjectName("StatusDot")
        self.lbl_status_dot.setStyleSheet("font-size: 16px; color: #757575;")
        
        self.lbl_status_tag = QLabel("系统状态：")
        self.lbl_status_tag.setObjectName("StatusTag")
        self.lbl_status_tag.setStyleSheet("font-size: 12px;")
        
        self.lbl_status_value = QLabel("待机中")
        self.lbl_status_value.setStyleSheet("font-size: 12px; font-weight: bold; color: #757575;")
        
        status_layout.addWidget(self.lbl_status_dot)
        status_layout.addWidget(self.lbl_status_tag)
        status_layout.addWidget(self.lbl_status_value)
        status_layout.addStretch()
        
        title_container.addWidget(self.lbl_title)
        title_container.addLayout(status_layout)
        header_layout.addLayout(title_container)

        header_layout.addStretch()

        self.btn_start = QPushButton("开始分析")
        self.btn_start.setObjectName("BtnStart")
        self.btn_reset = QPushButton("重置系统")
        self.btn_reset.setObjectName("BtnReset")
        self.btn_reset.setStyleSheet("background-color: #757575; color: white; border-radius: 4px; padding: 8px;") 
        header_layout.addWidget(self.btn_start)
        header_layout.addWidget(self.btn_reset)
        
        # 绑定信号
        self.btn_start.clicked.connect(self.toggle_analysis)
        self.btn_reset.clicked.connect(self.reset_system)

        # 主题切换下拉框
        header_layout.addSpacing(20)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        header_layout.addWidget(self.theme_combo)

        self.main_layout.addWidget(self.header_frame)

        # 2. 左右分栏布局 (Context)
        content_container = QWidget()
        content_layout = QHBoxLayout(content_container)
        content_layout.setContentsMargins(10, 10, 10, 10)
        
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧区域 - 视频/推理 Tab
        from ui.widgets.video_widget import VideoDisplayContainer
        self.left_tabs = QTabWidget()
        
        # 1. 实时分析主 Tab (整合所有源)
        self.tab_analysis = QWidget()
        self.analysis_layout = QVBoxLayout(self.tab_analysis)
        self.analysis_layout.setContentsMargins(5, 5, 5, 5)
        self.analysis_layout.setSpacing(5)
        
        # 顶部紧凑工具栏
        self.source_toolbar = QHBoxLayout()
        
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["本地视频", "实时摄像头", "网络流媒体"])
        self.source_type_combo.setFixedWidth(120)
        
        # 叠加式输入区 (Stacked Layout)
        from PySide6.QtWidgets import QStackedWidget
        self.source_input_stack = QStackedWidget()
        self.source_input_stack.setFixedHeight(35)
        
        # File Input
        self.page_file = QWidget()
        file_layout = QHBoxLayout(self.page_file)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_select_file = QPushButton("浏览...")
        self.btn_select_file.setFixedWidth(80)
        self.btn_select_file.clicked.connect(self.select_video_file)
        self.lbl_file_path = QLabel("未选择文件")
        self.lbl_file_path.setStyleSheet("color: #888; font-size: 11px;")
        file_layout.addWidget(self.btn_select_file)
        file_layout.addWidget(self.lbl_file_path)
        file_layout.addStretch()
        
        # Camera Input
        self.page_camera = QWidget()
        cam_layout = QHBoxLayout(self.page_camera)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.camera_combo.setFixedWidth(150)
        cam_layout.addWidget(self.camera_combo)
        cam_layout.addStretch()
        
        # Stream Input
        self.page_stream = QWidget()
        stream_layout = QHBoxLayout(self.page_stream)
        stream_layout.setContentsMargins(0, 0, 0, 0)
        self.edit_stream_url = QLineEdit()
        self.edit_stream_url.setPlaceholderText("rtsp://...")
        stream_layout.addWidget(self.edit_stream_url)
        
        self.source_input_stack.addWidget(self.page_file)
        self.source_input_stack.addWidget(self.page_camera)
        self.source_input_stack.addWidget(self.page_stream)
        
        self.source_type_combo.currentIndexChanged.connect(self.source_input_stack.setCurrentIndex)
        
        self.source_toolbar.addWidget(QLabel("输入源:"))
        self.source_toolbar.addWidget(self.source_type_combo)
        self.source_toolbar.addWidget(self.source_input_stack)
        
        self.video_container = VideoDisplayContainer()
        self.analysis_layout.addLayout(self.source_toolbar)
        self.analysis_layout.addWidget(self.video_container)
        
        self.tab_batch = QWidget()
        
        self.left_tabs.addTab(self.tab_analysis, "实时分析")
        self.left_tabs.addTab(self.tab_batch, "批量处理")
        self.content_splitter.addWidget(self.left_tabs)

        # 右侧区域 - 数据/统计 Tab
        from ui.widgets.dashboard_widget import DashboardWidget
        self.right_tabs = QTabWidget()
        self.dashboard = DashboardWidget()
        self.right_tabs.addTab(self.dashboard, "概览")
        self.right_tabs.addTab(QWidget(), "数据")
        self.right_tabs.addTab(QWidget(), "报告")
        self.right_tabs.addTab(QWidget(), "统计")
        self.right_tabs.addTab(QWidget(), "姿态")
        
        # 设置页面
        self.settings_tab = QWidget()
        settings_layout = QVBoxLayout(self.settings_tab)
        
        from PySide6.QtWidgets import QCheckBox, QTextEdit, QDoubleSpinBox, QFormLayout, QGroupBox
        self.chk_skeleton = QCheckBox("显示球员骨架")
        self.chk_skeleton.setChecked(True)
        self.chk_skeleton.toggled.connect(lambda v: setattr(self.video_container.display, 'show_skeletons', v))
        
        self.chk_ball = QCheckBox("显示羽毛球迹")
        self.chk_ball.setChecked(True)
        self.chk_ball.toggled.connect(lambda v: setattr(self.video_container.display, 'show_ball', v))
        
        self.chk_court = QCheckBox("显示球场轮廓")
        self.chk_court.setChecked(True)
        self.chk_court.toggled.connect(lambda v: setattr(self.video_container.display, 'show_court', v))
        
        settings_layout.addWidget(self.chk_skeleton)
        settings_layout.addWidget(self.chk_ball)
        settings_layout.addWidget(self.chk_court)

        # 算法阈值设置区
        algo_group = QGroupBox("模型算法阈值")
        algo_layout = QFormLayout(algo_group)
        
        self.spin_yolo_conf = QDoubleSpinBox()
        self.spin_yolo_conf.setRange(0.1, 1.0)
        self.spin_yolo_conf.setSingleStep(0.05)
        self.spin_yolo_conf.setValue(0.25)
        self.spin_yolo_conf.valueChanged.connect(self.update_algo_thresholds)
        
        self.spin_track_sens = QDoubleSpinBox()
        self.spin_track_sens.setRange(0.1, 1.0)
        self.spin_track_sens.setSingleStep(0.05)
        self.spin_track_sens.setValue(0.5)
        self.spin_track_sens.valueChanged.connect(self.update_algo_thresholds)

        self.spin_bst_conf = QDoubleSpinBox()
        self.spin_bst_conf.setRange(0.1, 1.0)
        self.spin_bst_conf.setSingleStep(0.05)
        self.spin_bst_conf.setValue(0.7)
        self.spin_bst_conf.valueChanged.connect(self.update_algo_thresholds)
        
        algo_layout.addRow("YOLO 置信度:", self.spin_yolo_conf)
        algo_layout.addRow("TrackNet 灵敏度:", self.spin_track_sens)
        algo_layout.addRow("BST 分类门限:", self.spin_bst_conf)
        
        settings_layout.addWidget(algo_group)
        settings_layout.addStretch()
        
        self.right_tabs.addTab(self.settings_tab, "设置")
        
        # 日志页面
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setObjectName("LogConsole")
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas;")
        self.right_tabs.addTab(self.log_output, "日志")
        
        self.content_splitter.addWidget(self.right_tabs)

        # 设置初始占比 (6.5:3.5) - 调宽右侧设置与仪表盘
        self.content_splitter.setStretchFactor(0, 65)
        self.content_splitter.setStretchFactor(1, 35)
        
        content_layout.addWidget(self.content_splitter)
        self.main_layout.addWidget(content_container)

        # 3. 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("系统就绪 | 设备正常")

    def select_video_file(self):
        """打开文件对话框选择视频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择羽毛球视频", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            self.lbl_file_path.setText(file_path.split("/")[-1])
            self.lbl_file_path.setToolTip(file_path)
            self.video_source = file_path
            self.append_log(f"已选择视频文件: {file_path}")
            self.update_app_status(AppStatus.READY)

    def closeEvent(self, event):
        """窗口关闭时的生命周期管理：安全释放资源"""
        reply = QMessageBox.question(self, '确认退出', "确定要停止分析并退出软件吗？",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 1. 强制停止引擎线程
            if self.engine_thread and self.engine_thread.isRunning():
                self.append_log("正在保存数据并释放显存...", "WARN")
                self.stop_analysis()
            
            # 2. 释放引擎单例资源 (显存清理关键)
            from core.engine.orchestrator import InferenceEngine
            engine = InferenceEngine()
            engine.close_source()
            
            self.append_log("系统安全退出")
            event.accept()
        else:
            event.ignore()

    def update_algo_thresholds(self):
        """实时更新后端模型阈值"""
        from core.engine.orchestrator import InferenceEngine
        engine = InferenceEngine() # 获取单例
        
        # 更新 YOLO 阈值
        if hasattr(engine.yolo_detector, 'conf_thres'):
            engine.yolo_detector.conf_thres = self.spin_yolo_conf.value()
        
        # 更新 TrackNet 灵敏度
        if hasattr(engine.ball_tracker, 'conf_thres'):
            engine.ball_tracker.conf_thres = self.spin_track_sens.value()
            
        # 更新 BST 分类门限
        if hasattr(engine.bst_generator, 'min_conf'):
            engine.bst_generator.min_conf = self.spin_bst_conf.value()
            
        self.append_log(f"参数已更新: YOLO={self.spin_yolo_conf.value():.2f}, "
                       f"Track={self.spin_track_sens.value():.2f}, "
                       f"BST={self.spin_bst_conf.value():.2f}")

    def apply_theme(self, theme_name):
        if theme_name == "Dark":
            self.setStyleSheet(ThemeManager.DARK_THEME)
        else:
            self.setStyleSheet(ThemeManager.LIGHT_THEME)
        logger.info(f"Theme switched to: {theme_name}")

    @Slot()
    def toggle_analysis(self):
        """切换分析状态"""
        if self.engine_thread and self.engine_thread.isRunning():
            self.stop_analysis()
        else:
            self.start_analysis()

    def start_analysis(self):
        """初始化并启动引擎线程"""
        # 性能保护：在打开新源前清理旧资源，防止显存碎片化
        if self.engine_thread and self.engine_thread.isRunning():
            self.stop_analysis()
            
        # 根据 StackedWidget 的当前索引判断输入源
        source_idx = self.source_input_stack.currentIndex()
        source_config = {}
        
        if source_idx == 0: # 本地文件
            if not hasattr(self, 'video_source'):
                QMessageBox.warning(self, "识别错误", "请先选择需要分析的视频文件！")
                return
            source_config = {"type": "file", "path": self.video_source}
        elif source_idx == 1: # 摄像头
            cam_idx = self.camera_combo.currentIndex()
            source_config = {"type": "camera", "index": cam_idx}
        elif source_idx == 2: # 网络流
            url = self.edit_stream_url.text().strip()
            if not url or not (url.startswith("rtsp://") or url.startswith("rtmp://") or url.startswith("http")):
                QMessageBox.critical(self, "连接错误", "输入的流地址格式无效，请检查！")
                return
            source_config = {"type": "rtsp", "url": url}
            
            # 显示连接等待框
            progress = QProgressDialog("正在连接流媒体服务器...", "取消", 0, 0, self)
            progress.setWindowTitle("网络连接中")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()
            time.sleep(0.5) 
            progress.close()

        if not self.engine_thread:
            self.engine_thread = EngineThread()
            self.engine_thread.frame_processed.connect(self.on_frame_processed)
            self.engine_thread.status_changed.connect(self.update_app_status)
        
        # 将配置注入引擎线程
        self.engine_thread.set_source(source_config)
        
        self.engine_thread.start()
        
        # UI 状态更新
        self.btn_start.setText("停止分析")
        self.btn_start.setStyleSheet("background-color: #D32F2F; color: white;")
        self.btn_select_file.setEnabled(False)
        self.left_tabs.setEnabled(False) # 分析时禁止切换 Tab
        
        self.update_app_status(AppStatus.ANALYZING)
        self.append_log(f"开始分析 - 源类型: {source_config['type']}")

    def stop_analysis(self):
        """停止引擎线程"""
        if self.engine_thread:
            self.engine_thread.stop()
            self.engine_thread.wait()
        
        # UI 按钮复原
        self.btn_start.setText("开始分析")
        self.btn_start.setStyleSheet("background-color: #1B5E20; color: white;")
        self.btn_select_file.setEnabled(True)
        self.left_tabs.setEnabled(True)
        
        self.update_app_status(AppStatus.READY)
        self.append_log("分析已停止")

    def reset_system(self):
        """重置系统状态：强制停止分析并清空所有数据"""
        # 强制停止并复位状态
        self.stop_analysis()
            
        # 1. 清空视频预览区
        self.video_container.display.set_frame(None)
        
        # 2. 清空右侧 KPI 与表格
        self.dashboard.card_fps.set_value("0.0")
        self.dashboard.card_infer.set_value("0.0")
        self.dashboard.card_status.set_value("待机中")
        self.dashboard.card_hits.set_value("0")
        self.dashboard.table.setRowCount(0)
        
        # 3. 清空日志
        self.log_output.clear()
        
        # 4. 状态复位
        self.update_app_status(AppStatus.IDLE)
        self.append_log("系统已重置")

    def append_log(self, message, level="INFO"):
        """推送到 UI 日志面板"""
        from datetime import datetime
        time_str = datetime.now().strftime("%H:%M:%S")
        color = "#ffffff"
        if level == "WARN": color = "#FFC107"
        elif level == "ERROR": color = "#F44336"
        
        html = f'<div style="margin-bottom: 2px;"><span style="color: #888888;">[{time_str}]</span> <span style="color: {color};">[{level}] {message}</span></div>'
        self.log_output.append(html)
        self.log_output.ensureCursorVisible()

    @Slot(object)
    def on_frame_processed(self, packet: FramePacket):
        """UI 更新插槽：接收处理后的数据包"""
        # UI 刷新率控制与采样丢帧
        now = time.time()
        if hasattr(self, '_last_render_time'):
            if now - self._last_render_time < 0.016: # 约 60fps 限制
                return
        self._last_render_time = now

        # 1. 更新视频预览
        self.video_container.update_frame(packet)
        
        # 2. 更新本地视频进度的 UI 反馈 (Slider & Timer)
        if 'progress' in packet.metadata:
            # 阻止信号触发递归
            self.video_container.slider.blockSignals(True)
            self.video_container.slider.setValue(int(packet.metadata['progress'] * 1000))
            self.video_container.slider.blockSignals(False)
            
            msec, total_msec = packet.metadata.get('time_info', (0, 0))
            def fmt(ms):
                s = int(ms / 1000)
                return f"{s//60:02d}:{s%60:02d}"
            self.video_container.lbl_time.setText(f"{fmt(msec)} / {fmt(total_msec)}")

        # 3. 更新右侧 KPI 仪表盘
        metrics = {
            'fps': packet.metadata.get('stream_fps', 0.0),
            'infer_fps': packet.metadata.get('infer_fps', 0.0),
            'rally': packet.metadata.get('is_rally', False),
            'hits': packet.metadata.get('hit_count', 0)
        }
        self.dashboard.update_metrics(metrics)
        
        # 4. 如果有新发生的击球事件，添加到动作列表
        if packet.metadata.get('is_new_event') and packet.stroke_action:
            action = packet.stroke_action
            msec, _ = packet.metadata.get('time_info', (0, 0))
            def fmt(ms):
                s = int(ms / 1000)
                return f"{s//60:02d}:{s%60:02d}"
                
            self.dashboard.add_action_row({
                'time': fmt(msec),
                'type': action.get('label', '未知'),
                'conf': action.get('conf', 0.0),
                'detail': f"Player {action.get('player_id')}"
            }, insert_at_top=True)
        # 4. 异常事件日志
        event_warn = packet.metadata.get('event_warning')
        if event_warn:
            self.append_log(event_warn, "WARN")

    @Slot(object)
    def update_app_status(self, status):
        """状态驱动逻辑：更新 UI 状态标签颜色和文字"""
        if isinstance(status, AppStatus):
            text, color = status.value
        elif isinstance(status, tuple):
            text, details = status
            color = "#FFA000" if "重试" in details or "断开" in text else "#388E3C"
            if details: text = f"{text} ({details})"
        else:
            text = str(status)
            color = "#757575"
            
        self.lbl_status_value.setText(text)
        self.lbl_status_value.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {color};")
        self.lbl_status_dot.setStyleSheet(f"font-size: 16px; color: {color};")
        self.status_bar.showMessage(f"系统状态: {text}")
        self.status_bar.showMessage(f"系统状态切换至: {text}")

if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()
    app = QApplication(sys.argv)
    window = BadmintonMainWindow()
    window.show()
    sys.exit(app.exec())
