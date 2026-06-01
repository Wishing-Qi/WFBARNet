import numpy as np
from typing import List, Optional, Tuple
from common.packet import FramePacket
from utils.logger import logger

class HitTrigger:
    """
    基于矢量夹角和人球距离的击球触发器。
    用于检测羽毛球轨迹的剧烈转向（击球瞬间）并触发动作识别。
    """
    def __init__(self, angle_threshold: float = 90.0, dist_threshold_meters: float = 1.5):
        self.angle_threshold = angle_threshold
        self.dist_threshold_meters = dist_threshold_meters
        self.ball_history: List[Tuple[float, float]] = [] # 存储物理坐标 (x, y) 米
        self.pixel_history: List[Tuple[float, float]] = [] # 兜底像素坐标

    def check_hit(self, packet: FramePacket, mapper) -> bool:
        """
        检查当前帧是否发生击球事件。
        mapper: 用于像素到物理坐标转换的 CourtMapper
        """
        if packet.ball_coord is None:
            return False

        # 1. 坐标转换：像素 -> 物理 (米)，若不可用则回退到像素空间
        real_pos = mapper.pixel_to_real(*packet.ball_coord)
        use_pixel = real_pos is None

        if use_pixel:
            self.pixel_history.append(packet.ball_coord[:2])
            if len(self.pixel_history) > 3:
                self.pixel_history.pop(0)
            if len(self.pixel_history) < 3:
                return False
        else:
            self.ball_history.append(real_pos)
            if len(self.ball_history) > 3:
                self.ball_history.pop(0)
            if len(self.ball_history) < 3:
                return False

        # 2. 计算矢量的夹角
        # v1: p(n-2) -> p(n-1)
        # v2: p(n-1) -> p(n)
        p1, p2, p3 = self.pixel_history if use_pixel else self.ball_history
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 0.01 or norm2 < 0.01:
            return False

        # 计算夹角 (余弦定理)
        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        # 限制范围防止数值误差
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))

        # 3. 检查人球距离 (物理距离 / 像素距离)
        min_dist = float('inf')
        for skel in packet.skeletons:
            # 使用关键点(如手腕)或 bbox 中心
            bbox = skel.get("bbox", [0, 0, 0, 0])
            p_center_pixel = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
            if use_pixel:
                d = np.sqrt((p_center_pixel[0]-p3[0])**2 + (p_center_pixel[1]-p3[1])**2)
                min_dist = min(min_dist, d)
            else:
                p_real = mapper.pixel_to_real(*p_center_pixel)
                if p_real:
                    d = np.sqrt((p_real[0]-real_pos[0])**2 + (p_real[1]-real_pos[1])**2)
                    min_dist = min(min_dist, d)

        # 4. 综合判定：矢量夹角大 (剧烈转向/反向) 且 离人近
        # 注意：顺滑飞行时 angle 接近 0，击球时 angle 会很大 (朝向反转)
        if use_pixel:
            # 像素兜底阈值，适配未标定时的击球统计
            if angle > self.angle_threshold and min_dist < 120:
                logger.info(f"Hit Detected (pixel)! Angle: {angle:.1f}deg, MinDist: {min_dist:.1f}px")
                return True
            return False

        if angle > self.angle_threshold and min_dist < self.dist_threshold_meters:
            logger.info(f"Hit Detected! Angle: {angle:.1f}deg, MinDist: {min_dist:.2f}m")
            return True

        return False
