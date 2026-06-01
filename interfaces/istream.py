from abc import ABC, abstractmethod
from common.packet import FramePacket

class IStreamPlugin(ABC):
    """
    所有流插件必须实现的接口。
    """
    @abstractmethod
    def connect(self, source: str) -> bool:
        """连接视频源 (文件路径, RTSP URL, 或 WebRTC 句柄)"""
        pass

    @abstractmethod
    def read(self) -> FramePacket:
        """读取并封装下一帧数据包"""
        pass

    @abstractmethod
    def release(self):
        """释放资源"""
        pass

    @property
    @abstractmethod
    def is_opened(self) -> bool:
        """检查流是否已正常打开"""
        pass
