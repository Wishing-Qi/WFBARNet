import cv2
import time
import numpy as np
from interfaces.istream import IStreamPlugin
from common.packet import FramePacket

class RTSPStreamPlugin(IStreamPlugin):
    def __init__(self):
        self.cap = None
        self.url = ""
        self._is_opened = False
        self.last_frame_time = 0

    def connect(self, source: str) -> bool:
        self.url = source
        self.cap = cv2.VideoCapture(source)
        self._is_opened = self.cap.isOpened()
        if self._is_opened:
            self.last_frame_time = time.time()
        return self._is_opened

    def read(self) -> FramePacket:
        if not self.cap or not self._is_opened:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None

        self.last_frame_time = time.time()
        
        # 封装基础 Packet
        packet = FramePacket(
            frame_id=int(time.time() * 1000), # 临时 ID
            timestamp=time.time() * 1000,
            source_id="rtsp",
            image=frame
        )
        return packet

    def release(self):
        if self.cap:
            self.cap.release()
        self._is_opened = False

    @property
    def is_opened(self) -> bool:
        return self._is_opened
