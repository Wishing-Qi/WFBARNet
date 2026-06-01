import cv2
import time
from interfaces.istream import IStreamPlugin
from common.packet import FramePacket

class FileStreamPlugin(IStreamPlugin):
    """
    本地视频文件流插件示例。
    """
    PLUGIN_ID = "file"

    def __init__(self):
        self.cap = None
        self.source = ""
        self.frame_count = 0

    def connect(self, source: str) -> bool:
        """
        source: 视频文件路径
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            return False
        self.frame_count = 0
        return True

    def read(self) -> FramePacket:
        if not self.is_opened:
            return None

        success, frame = self.cap.read()
        if not success:
            return None

        # 获取当前帧在视频中的百分比 (0.0 - 1.0)
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        curr_frame_idx = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        progress = curr_frame_idx / total_frames if total_frames > 0 else 0
        
        # 记录时间
        msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        total_msec = total_frames / self.cap.get(cv2.CAP_PROP_FPS) * 1000 if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 0

        self.frame_count += 1
        timestamp = time.time()

        packet = FramePacket(
            frame_id=self.frame_count,
            timestamp=timestamp,
            source_id="local_file",
            image=frame
        )
        # 将进度和时间信息存入元数据
        packet.metadata['progress'] = progress 
        packet.metadata['time_info'] = (msec, total_msec)
        
        return packet

    def release(self):
        if self.cap:
            self.cap.release()

    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
