import numpy as np
import torch
from collections import deque
from typing import Dict, List, Optional, Tuple
from common.packet import FramePacket
from utils.logger import logger
from utils.config_loader import cfg

class BSTFeatureGenerator:
    """
    BST 动作识别特征生成器。
    为每个球员维护时序缓冲区，提取骨架与球迹相对特征。
    """
    def __init__(self):
        # 从配置获取窗口长度，默认为 31
        self.window_size = cfg.get("BST.window_size", 31)
        self.player_buffers = {} # track_id -> deque of frame data
        
    def update(self, packet: FramePacket):
        """
        处理当前帧，更新所有已知球员的缓冲区并检查触发逻辑。
        """
        ball_pos = packet.ball_coord # [x, y] 像素坐标
        
        for skel in packet.skeletons:
            track_id = skel.get("player_id")
            if track_id is None or track_id == -1:
                continue
                
            if track_id not in self.player_buffers:
                self.player_buffers[track_id] = deque(maxlen=self.window_size)
            
            # 1. 提取并归一化当前帧特征
            frame_data = self._extract_features(skel, ball_pos)
            self.player_buffers[track_id].append(frame_data)
            
            # 2. 检查触发动作识别逻辑
            if len(self.player_buffers[track_id]) == self.window_size:
                # 检查“击球候选”：球离人近或球速发生变化（简单版本：距离阈值）
                if self._is_hitting_candidate(skel, ball_pos):
                    input_tensor = self._prepare_tensor(track_id)
                    # 将生成的特征张量挂载到骨架信息中
                    skel["bst_input"] = input_tensor

    def _extract_features(self, skel: Dict, ball_pos: Optional[List[float]]) -> Dict:
        """
        提取并归一化单帧特征。
        - 骨架：以盆骨为中心
        - 球迹：相对于持拍手腕的相对坐标
        """
        kpts = np.array(skel.get("keypoints", [])).reshape(-1, 3) # [17, 3]
        
        # 归一化原点：盆骨中点 (COCO: 11-L_Hip, 12-R_Hip)
        if kpts[11, 2] > 0.1 and kpts[12, 2] > 0.1:
            root = (kpts[11, :2] + kpts[12, :2]) / 2
        else:
            bbox = skel.get("bbox", [0, 0, 0, 0])
            root = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            
        rel_kpts = kpts[:, :2] - root
        
        # 人球相对位移 (相对于主导手腕，假设右手 10)
        wrist_pos = kpts[10, :2] if kpts[10, 2] > 0.1 else kpts[9, :2]
        rel_ball = np.array([0.0, 0.0])
        if ball_pos is not None:
            # 只取坐标前两项 [x, y]，忽略置信度
            rel_ball = np.array(ball_pos[:2]) - wrist_pos

        return {
            "rel_kpts": rel_kpts, # (17, 2)
            "rel_ball": rel_ball  # (2,)
        }

    def _is_hitting_candidate(self, skel: Dict, ball_pos: Optional[List[float]]) -> bool:
        """
        判断当前是否可能发生击球行为。
        """
        if ball_pos is None:
            return False
            
        bbox = skel.get("bbox", [0, 0, 0, 0])
        # 扩展 bbox 范围检查球是否在球员附近
        center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        dist = np.sqrt((center[0]-ball_pos[0])**2 + (center[1]-ball_pos[1])**2)
        
        return dist < 200 # 像素距离阈值，视分辨率调整

    def _prepare_tensor(self, track_id: int) -> torch.Tensor:
        """
        转换为 (1, T, 36) 张量。
        """
        buffer = self.player_buffers[track_id]
        kpts_seq = np.array([f["rel_kpts"] for f in buffer]).reshape(self.window_size, -1)
        ball_seq = np.array([f["rel_ball"] for f in buffer])
        
        combined = np.concatenate([kpts_seq, ball_seq], axis=1) # (T, 36)
        return torch.from_numpy(combined).float().unsqueeze(0)
