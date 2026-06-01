from PySide6.QtCore import QThread, Signal
from common.packet import FramePacket
from core.engine.orchestrator import InferenceEngine
import time

class EngineThread(QThread):
    """
    后端引擎运行线程，负责将推断结果通过信号传递给 UI。
    """
    frame_processed = Signal(object)  # 发送 FramePacket 对象
    status_changed = Signal(object)   # 发送 (AppStatus, extra_msg)
    stream_error = Signal(str)        # 发送错误消息

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = InferenceEngine()
        self.is_running = False
        self.source_config = None
        self.retry_count = 0

    def set_source(self, config):
        self.source_config = config

    def stop(self):
        """停止引擎循环"""
        self.is_running = False
        self.engine.running = False

    def run(self):
        """线程主逻辑"""
        self.is_running = True
        self.engine.running = True
        
        import cv2
        import numpy as np

        last_frame_time = time.time()
        backoff_time = 1.0 # 初始重连等待时间

        while self.is_running:
            # 1. 尝试获取/加载源
            if not self.engine.has_source():
                self.status_changed.emit(("正在连接...", ""))
                success = self.engine.load_source(self.source_config)
                if not success:
                    self.retry_count += 1
                    wait_time = min(backoff_time * (2 ** (self.retry_count - 1)), 60)
                    self.status_changed.emit(("连接失败", f"重试中({self.retry_count}), 等待{wait_time:.1f}s"))
                    # 模拟等待
                    for _ in range(int(wait_time * 10)):
                        if not self.is_running: break
                        time.sleep(0.1)
                    continue
                else:
                    self.retry_count = 0
                    self.status_changed.emit(("连接成功", "数据流正常"))
                    last_frame_time = time.time()

            # 2. 读取数据包
            packet = self.engine.get_next_packet()
            
            if packet:
                last_frame_time = time.time()
                # 移除预览图缩放逻辑，由 VideoWidget 的 paintEvent 统一处理比例映射
                # 这样可以保持检测坐标与图像尺寸的一致性
                self.frame_processed.emit(packet)
            else:
                # 3. 心跳检测 (3秒无帧)
                if time.time() - last_frame_time > 3.0:
                    if self.source_config and self.source_config.get('type') in ['rtsp', 'camera']:
                        self.status_changed.emit(("源已断开", "尝试自动重连..."))
                        self.engine.close_source() # 触发重连逻辑
                        last_frame_time = time.time()
                
                time.sleep(0.01)

        self.engine.close_source()
        self.is_running = False
