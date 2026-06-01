from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

@dataclass
class FramePacket:
    """
    核心数据包标准。
    包含元数据、原始图像以及各阶段模型的处理结果。
    """
    frame_id: int
    timestamp: float
    source_id: str
    image: np.ndarray             # 原始图像 (OpenCV 格式: HWC, BGR)
    
    # --- 统一数据字段 ---
    ball_xy: Optional[Tuple[int, int]] = None  # 球的 (x, y) 坐标
    ball_coord: Optional[tuple] = None         # (x, y, conf) 兼容旧版
    
    # 球员列表: [{player_id: int, bbox: [x1, y1, x2, y2, score], keypoints: np.ndarray, stroke_action: str}]
    players: List[Dict[str, Any]] = field(default_factory=list)
    
    # 为了保持兼容性保留 skeletons 字段
    skeletons: List[Dict] = field(default_factory=list) 
    
    # 场线坐标 (例如角点或特征点)
    court_points: List[Tuple[int, int]] = field(default_factory=list)
    court_info: Dict = field(default_factory=dict)     
    
    # 动作/击球识别结果
    stroke_action: Optional[str] = None   
    
    # --- 扩展接口 ---
    # 预留扩展字段，用于存储插件或自定义模块的中间结果
    metadata: Dict[str, Any] = field(default_factory=dict) 

    def update_result(self, key: str, value: Any):
        """通用结果更新方法"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.metadata[key] = value

    def get_result(self, key: str, default: Any = None) -> Any:
        """获取结果，优先查找属性，其次查找 metadata"""
        return getattr(self, key, self.metadata.get(key, default))
