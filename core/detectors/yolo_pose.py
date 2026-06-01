import cv2
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from core.base_model import BaseModel

class YOLOPoseDetector(BaseModel):
    """
    YOLO-Pose 检测器包装类。
    负责检测球员（Bbox）及其骨骼关键点，并扩展检测球场角点。
    """
    def __init__(self, model_path: str, device: str = "cuda", use_trt: bool = False):
        super().__init__(model_path, device)
        self.use_trt = use_trt
        self.conf_thres = 0.25 # 默认阈值
        self.load()
        
        # COCO 17 个关键点定义
        self.keypoint_mapping = {
            "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
            "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
            "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
            "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
        }

    def load(self):
        """
        加载模型并移动到指定设备。
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            # 预热一帧
            # self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")

    def predict(self, image: np.ndarray) -> Tuple[List[Dict], List[Tuple]]:
        """
        执行推理。
        """
        if self.model is None:
            return [], []

        h_orig, w_orig = image.shape[:2]
        
        # 执行推理，禁用打印输出
        results = self.model.predict(image, conf=self.conf_thres, verbose=False)
        
        players_results = []
        court_corners = [] 

        if results and len(results) > 0:
            res = results[0]
            # 提取检测框和关键点
            if res.boxes is not None and res.keypoints is not None:
                boxes = res.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
                scores = res.boxes.conf.cpu().numpy() # [conf]
                kpts = res.keypoints.data.cpu().numpy() # [N, 17, 3] (x, y, conf)
                
                for i in range(len(boxes)):
                    players_results.append({
                        "bbox": np.append(boxes[i], scores[i]), # [x1, y1, x2, y2, conf]
                        "keypoints": kpts[i]
                    })
        
        # 暂时没有单独的球场检测模型，我们可以根据 YOLO 的结果或者特定的坐标逻辑
        # 这里预留空列表，由手动标定填充
        return players_results, court_corners
