import cv2
import torch
import numpy as np
from collections import deque
from typing import Optional, Tuple, List
from core.base_model import BaseModel
from .tracknet_v3_model import TrackNetV2

class TrackNetWrapper(BaseModel):
    """
    TrackNetV3 球迹追踪包装类。
    输入：当前连续 3 帧图像。
    输出：球在当前帧的归一化像素坐标 (x, y) 及置信度。
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(model_path, device)
        self.input_width = 512
        self.input_height = 288
        
        # 帧缓冲区，用于存储连续 3 帧
        self.frame_buffer = deque(maxlen=3)
        self.load()

    def load(self):
        """
        加载模型定义及权重。
        """
        self.model = TrackNetV2(in_dim=9, out_dim=3)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        self.model.to(self.device).eval()

    def preprocess(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        预处理逻辑：缩放和拼接
        frames: 3 帧 BGR 图像列表
        """
        processed_frames = []
        for frame in frames:
            img = cv2.resize(frame, (self.input_width, self.input_height))
            img = img.astype(np.float32) / 255.0
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
            processed_frames.append(img)
        
        # 拼接 3 帧 (9, 288, 512)
        input_tensor = np.concatenate(processed_frames, axis=0)
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device)
        return input_tensor

    def predict(self, current_frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        输入当前帧，内部维护 deque，当满 3 帧时输出归一化的预测结果与置信度。
        """
        self.frame_buffer.append(current_frame)
        
        if len(self.frame_buffer) < 3:
            return None
        
        # 提取 3 帧进行推理
        input_data = self.preprocess(list(self.frame_buffer))
        
        with torch.no_grad():
            output = self.model(input_data)
            output = output.cpu().numpy()[0] # (3, 288, 512)
            
        # 提取最后一帧对应的热力图 (即当前帧)
        heatmap = output[2] 
        
        # 寻找最大值点
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
        
        if max_val < 0.5: # 阈值过滤
            return None
            
        # 归一化坐标输出 (x/w, y/h)
        norm_x = max_loc[0] / self.input_width
        norm_y = max_loc[1] / self.input_height
        
        return (norm_x, norm_y, float(max_val))
