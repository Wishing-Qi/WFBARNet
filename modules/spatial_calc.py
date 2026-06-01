import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from utils.config_loader import cfg
from utils.logger import logger
from utils.math_physics import ShuttlePhysics

class CourtMapper:
    """
    空间几何映射类：处理 2D 像素坐标与物理平面坐标之间的转换。
    """
    def __init__(self):
        # 从配置加载标准场地参数
        self.court_width = cfg.get("BadmintonCourt.width", 6.1)
        self.court_length = cfg.get("BadmintonCourt.length", 13.4)
        
        # 标准场地 4 个角点 (米制)
        self.src_real_points = np.array([
            cfg.get("BadmintonCourt.target_points.top_left", [0, 13.4]),
            cfg.get("BadmintonCourt.target_points.top_right", [6.1, 13.4]),
            cfg.get("BadmintonCourt.target_points.bottom_left", [0, 0]),
            cfg.get("BadmintonCourt.target_points.bottom_right", [6.1, 0])
        ], dtype=np.float32)

        self.homography_matrix = None
        self.last_valid_matrix = None
        self.fps = 30 # 默认，运行时应由 Engine 更新

    def update_homography(self, corner_points: List[Tuple[float, float]]):
        """
        更新单应性变换矩阵。
        corner_points: [(x,y), ...] 顺序必须与配置中的 TL, TR, BL, BR 一致。
        """
        if not corner_points or len(corner_points) < 4:
            if self.last_valid_matrix is not None:
                logger.warning("Court corners missing/invalid, using last valid homography matrix.")
                self.homography_matrix = self.last_valid_matrix
            return False

        src_pixel_points = np.array(corner_points, dtype=np.float32)
        
        # 计算从像素到物理平面的变换
        matrix, _ = cv2.findHomography(src_pixel_points, self.src_real_points)
        
        if matrix is not None:
            self.homography_matrix = matrix
            self.last_valid_matrix = matrix
            return True
        return False

    def pixel_to_real(self, x: float, y: float, *args) -> Optional[Tuple[float, float]]:
        """
        将像素坐标转换为场地物理坐标 (x_meter, y_meter)。
        支持传入额外的参数（如置信度）而不报错。
        """
        if self.homography_matrix is None:
            return None
        
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))

    def calculate_speed(self, p1_real: Tuple[float, float], p2_real: Tuple[float, float], dt: float) -> float:
        """
        计算两点间的物理速度 (m/s)。
        """
        dist = np.sqrt((p1_real[0] - p2_real[0])**2 + (p1_real[1] - p2_real[1])**2)
        return dist / dt if dt > 0 else 0.0

class KinematicsAnalyzer:
    """
    运动学参数计算工具。
    """
    def __init__(self, fps=30):
        self.player_history = {} # track_id -> {"last_pos": (x,y), "total_dist": 0.0, "current_speed": 0.0}
        self.ball_history = []  # List of real_coords (2D: x, y)
        self.ball_history_3d = [] # List of estimated 3D coords (x, y, z)
        self.shuttle_physics = ShuttlePhysics(fps=fps)
        self.fps = fps

    def update_player_metrics(self, track_id: int, current_pixel_pos: Tuple[float, float], mapper: CourtMapper):
        """
        更新球员跑动指标。
        """
        real_pos = mapper.pixel_to_real(*current_pixel_pos)
        if real_pos is None:
            return None

        dt = 1.0 / self.fps
        if track_id not in self.player_history:
            self.player_history[track_id] = {"last_pos": real_pos, "total_dist": 0.0, "current_speed": 0.0}
            return self.player_history[track_id]

        hist = self.player_history[track_id]
        
        # 计算单帧位移和速度
        speed = mapper.calculate_speed(hist["last_pos"], real_pos, dt)
        dist = np.sqrt((real_pos[0] - hist["last_pos"][0])**2 + (real_pos[1] - hist["last_pos"][1])**2)
        
        # 简单平滑过滤 (针对 YOLO 检测抖动引起的瞬时速度过快)
        if speed < 12.0: # 正常球员跑动速度限制 (~12m/s)
            hist["total_dist"] += dist
            hist["current_speed"] = speed
            hist["last_pos"] = real_pos
        
        return hist

    def update_ball_speed(self, current_pixel_pos: Tuple[float, float], mapper: CourtMapper) -> float:
        """
        计算羽毛球物理速度 (km/h)，包含 3D 轨迹平滑与高度估计。
        """
        real_pos_2d = mapper.pixel_to_real(*current_pixel_pos)
        if real_pos_2d is None:
            return 0.0
            
        self.ball_history.append(real_pos_2d)
        if len(self.ball_history) > 5:
            self.ball_history.pop(0)
            
        # 进行 3D 估计与平滑
        current_3d = self.shuttle_physics.estimate_3d_trajectory(self.ball_history)
        self.ball_history_3d.append(current_3d)
        if len(self.ball_history_3d) > 2:
            self.ball_history_3d.pop(0)

        if len(self.ball_history_3d) == 2:
            speed_kmh = self.shuttle_physics.calculate_3d_speed(
                self.ball_history_3d[0], 
                self.ball_history_3d[1]
            )
            return speed_kmh
        return 0.0
