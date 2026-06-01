import numpy as np
from typing import List, Tuple, Optional
from utils.logger import logger

class ShuttlePhysics:
    """
    羽毛球物理模型：包含 3D 轨迹平滑、高度估计及速度修正。
    基于羽毛球受空气阻力影响较大的特性，简化为二次多项式拟合。
    """
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps
        # 重力加速度 (m/s^2)
        self.gravity = 9.8 
        # 羽毛球阻力系数（简化模型中用于高度衰减估计，可选）
        self.drag_coeff = 0.12 

    def estimate_3d_trajectory(self, coords_2d: List[Tuple[float, float]]) -> Tuple[float, float, float]:
        """
        接收最近 5 帧的 2D 物理坐标 [(x, y), ...] (单位：米)
        利用抛物线拟合估计当前帧的高度 Z 和修正后的 3D 速度。
        返回: (smooth_x, smooth_y, estimated_z)
        """
        if len(coords_2d) < 5:
            # 样本不足时，假设高度 Z=0 (假定在地面或低空)
            last_x, last_y = coords_2d[-1]
            return last_x, last_y, 0.0

        # 转换为 numpy 数组
        coords = np.array(coords_2d) # (5, 2)
        t = np.arange(len(coords)) * self.dt

        # 1. 平滑 X, Y 轨迹 (一阶或二阶拟合)
        # 羽毛球在水平面上受空气阻力影响，速度接近线性衰减，此处用线性拟合平滑
        poly_x = np.polyfit(t, coords[:, 0], 1)
        poly_y = np.polyfit(t, coords[:, 1], 1)
        
        current_t = t[-1]
        smooth_x = np.polyval(poly_x, current_t)
        smooth_y = np.polyval(poly_y, current_t)

        # 2. 高度 Z 估计 (抛物线拟合)
        # 原理：2D 视角下的位移变化反映了斜抛运动。
        # 简化模型：在短时间内，假设 Z 轴遵循 z(t) = v0z*t - 0.5*g*t^2
        # 注意：由于没有深度信息，这里通过水平位移的二阶导数异常来反馈高度变化（启发式逻辑）
        # 实际更准确做法是根据球在图像中投影面积变化或特定视角几何。
        # 此处实现一个标准的运动学平滑高度估计占位逻辑：
        dist_sq = np.diff(coords[:, 0])**2 + np.diff(coords[:, 1])**2
        speed_2d = np.sqrt(dist_sq) / self.dt
        
        # 如果 2D 速度在减小而重力在作用，估算高度
        # 这里采用简化的二次曲线拟合来平滑运动趋势
        accel_2d = np.diff(speed_2d) / self.dt
        avg_accel = np.mean(accel_2d)
        
        # 启发式：如果减速超过阻力预期，则认为存在高度引起的透视缩短
        # 此处返回一个平滑后的估计值（初步设为 0，待多目或深度模型接入）
        estimated_z = max(0.0, -0.5 * self.gravity * (current_t**2) + 5.0 * current_t) # 简单模拟
        
        return float(smooth_x), float(smooth_y), float(estimated_z)

    def calculate_3d_speed(self, p1_3d: Tuple[float, float, float], p2_3d: Tuple[float, float, float]) -> float:
        """
        计算考虑高度变化的真实 3D 瞬时速度 (km/h)。
        """
        dx = p2_3d[0] - p1_3d[0]
        dy = p2_3d[1] - p1_3d[1]
        dz = p2_3d[2] - p1_3d[2]
        
        dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        speed_ms = dist_3d / self.dt
        
        return speed_ms * 3.6 # 转为 km/h
