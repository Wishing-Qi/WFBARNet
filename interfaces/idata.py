from abc import ABC, abstractmethod
from typing import Any, Dict

class IDataPacket(ABC):
    """
    统一数据包的抽象接口定义，确保所有数据包实现都具备基础元数据。
    """
    @property
    @abstractmethod
    def frame_id(self) -> int:
        pass

    @property
    @abstractmethod
    def timestamp(self) -> float:
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        pass
