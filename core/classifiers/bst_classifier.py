import torch
import numpy as np
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from .model.bst import BST
from .bst_utils import get_bone_pairs, create_bones, interpolate_joints, get_stroke_types
from utils.logger import logger

class BSTClassifier:
    """
    BST (Badminton Stroke-type Transformer) 动作识别包装类。
    输入：连续帧的球员骨骼序列和球迹序列。
    输出：识别出的动作类型（Smash, Drop 等）。
    """
    def __init__(self, model_path: str, seq_len: int = 30, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.n_joints = 17
        self.bone_pairs = get_bone_pairs()
        self.n_bones = len(self.bone_pairs)
        
        # 修正参数以匹配权重文件:
        in_dim = 72
        self.model = BST(
            in_dim=in_dim,
            n_class=25, 
            seq_len=seq_len,
            depth_tem=2,
            depth_inter=1
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        with torch.no_grad():
            try:
                # 严格模式设为 False 以忽略意外的 mlp_clean 键
                self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint, strict=False)
            except Exception as e:
                print(f"Warning during model loading: {e}")
                
        self.model.to(self.device).eval()
        
        self.stroke_types = get_stroke_types()
        
        # 维护每个球员的历史轨迹：player_id -> 历史数据队列
        # 每个数据点：(skeleton_2d, ball_pos_2d, player_pos_2d)
        self.history = {} 

    def update_history(self, player_id: int, skeleton: np.ndarray, ball_coord: Optional[tuple]):
        """
        更新特定球员的历史记录。
        skeleton: (17, 2), 归一化后的坐标
        ball_coord: (x, y, conf)
        """
        if player_id not in self.history:
            self.history[player_id] = deque(maxlen=self.seq_len)
            
        # 计算球员中心位置 (通常用两胯的中点或者 bbox 中心)
        player_pos = np.mean(skeleton, axis=0) # 简易处理
        
        # 处理球的位置 (如果当前帧没检测到球，使用 (0,0))
        shuttle_pos = np.array([0.0, 0.0], dtype=np.float32)
        if ball_coord is not None:
            # 这里需要与模型训练时的归一化逻辑对齐
            shuttle_pos = np.array(ball_coord[:2], dtype=np.float32)
            
        self.history[player_id].append({
            'joints': skeleton,
            'pos': player_pos,
            'shuttle': shuttle_pos
        })

    def predict(self, player_id: int) -> Optional[str]:
        """
        对特定球员的一段序列进行动作识别。
        """
        if player_id not in self.history or len(self.history[player_id]) < self.seq_len:
            return None
            
        data = list(self.history[player_id])
        # 修正特征构建：匹配 in_dim = 72 (Jobs + Bones) * 2
        # (F, J, 2)
        joints = np.array([d['joints'] for d in data]) # (31, 17, 2)
        
        # 简单计算骨骼特征 (F, 19, 2)
        bones = []
        for f in range(self.seq_len):
            f_bones = []
            for p1, p2 in self.bone_pairs:
                f_bones.append(joints[f, p2] - joints[f, p1])
            bones.append(f_bones)
        bones = np.array(bones) # (31, 19, 2)
        
        # 拼接 (T, 17+19, 2) -> (T, 72)
        feat = np.concatenate([joints, bones], axis=1).reshape(self.seq_len, -1).astype(np.float32)

        # 组装为 2 人输入 (第二人补零)
        zeros_feat = np.zeros_like(feat)
        feat_2p = np.stack([feat, zeros_feat], axis=1)  # (T, 2, 72)

        # 之前 history 中存储的 player_pos 和 shuttle
        pos = np.array([d['pos'] for d in data], dtype=np.float32)
        zeros_pos = np.zeros_like(pos)
        pos_2p = np.stack([pos, zeros_pos], axis=1)  # (T, 2, 2)

        shuttle = np.array([d['shuttle'] for d in data], dtype=np.float32)  # (T, 2)

        feat_tensor = torch.from_numpy(feat_2p).unsqueeze(0).float().to(self.device)
        pos_tensor = torch.from_numpy(pos_2p).unsqueeze(0).float().to(self.device)
        shuttle_tensor = torch.from_numpy(shuttle).unsqueeze(0).float().to(self.device)
        video_len_tensor = torch.tensor([self.seq_len]).to(self.device)

        with torch.no_grad():
            try:
                logits = self.model(feat_tensor, shuttle_tensor, pos_tensor, video_len_tensor)
                pred_idx = torch.argmax(logits, dim=1).item()
                if isinstance(self.stroke_types, dict):
                    return self.stroke_types.get(pred_idx, f"Unknown_{pred_idx}")
                if 0 <= pred_idx < len(self.stroke_types):
                    return self.stroke_types[pred_idx]
                return f"Unknown_{pred_idx}"
            except Exception as e:
                logger.error(f"BST predict failed: {e}")
                return None
