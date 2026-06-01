from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch

class BaseModel(ABC):
    """
    模型抽象基类，规范初始化、预测和后端配置。
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        # 兼容 int 和 str 类型的 device
        target_device = device
        if isinstance(device, int):
            target_device = f"cuda:{device}"
            
        self.device = torch.device(target_device if torch.cuda.is_available() else "cpu")
        print(f"[Model Init] {self.__class__.__name__} loading on {self.device}")
        self.model = None
        self._setup_backend()

    def _setup_backend(self):
        """
        自动检测推理引擎。
        优先使用 TensorRT (适配 40 系列显卡)，其次是 CUDA。
        """
        if torch.cuda.is_available():
            # 记录后端日志
            pass

    @abstractmethod
    def load(self):
        """加载权重"""
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """执行推理"""
        pass
