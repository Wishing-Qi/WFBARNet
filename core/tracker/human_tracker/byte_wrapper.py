import numpy as np
from typing import List, Dict, Any, Optional
from .byte_tracker import BYTETracker

class ByteTrackWrapper:
    """
    ByteTrack 球员追踪包装类。
    输入：每帧的检测框 (Bbox + Score)。
    输出：带有 track_id 的追踪结果，并同步到 FramePacket。
    """
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        # 初始化原生的 BYTETracker
        # 参数根据羽毛球场景可微调
        class Args:
            def __init__(self):
                self.track_thresh = track_thresh
                self.track_buffer = track_buffer
                self.match_thresh = match_thresh
                self.mot20 = False # 羽毛球场通常不属于密集人群场景
        
        self.args = Args()
        self.tracker = BYTETracker(self.args, frame_rate=frame_rate)

    def update(self, detections: np.ndarray, img_info: tuple) -> List[Any]:
        """
        更新追踪状态
        detections: shape (N, 5), 格式 [x1, y1, x2, y2, score]
        img_info: (height, width)
        """
        if detections is None or len(detections) == 0:
            # 即使没有检测到，也需要调用 update 以维持卡尔曼滤波预测
            detections = np.empty((0, 5))
            
        # ByteTrack 的 update 期望输入是检测框和图像信息
        # 返回的是当前帧处于 Tracked 状态的 STrack 对象列表
        online_targets = self.tracker.update(detections, img_info, img_info)
        
        return online_targets

    def apply_to_packet(self, online_targets: List[Any], packet_skeletons: List[Dict]):
        """
        将追踪到的 ID 应用到 FramePacket 的 skeletons 中。
        这里假设 YOLO 检测结果已经初始化了 skeletons 的 bbox。
        """
        # 建立位置索引或通过 IOU 匹配 ID（ByteTrack 返回的是独立的对象）
        # 简单实现：将追踪到的目标信息转换后供后续使用
        for target in online_targets:
            tlwh = target.tlwh
            track_id = target.track_id
            
            # 匹配逻辑：在 skeletons 中找到与 tlwh (top-left, width, height) 最接近的项
            # 或者在 Engine 层先进行检测，再由 wrapper.update，最后统一写入
            # 这里我们为 packet 提供一个更新列表的方法
            found = False
            for skel in packet_skeletons:
                # 如果 bbox 已经存在且与 tlwh 匹配度高，则分配 ID
                # (实际场景中通常在检测后立即追踪，两者顺序需要协调)
                pass 
                
        return online_targets
