import threading
import queue
import time
import cv2
import numpy as np
import sys
import os
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional

# 自动定位项目根目录，确保 interfaces, common 等模块可导入
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.append(ROOT)

from interfaces.istream import IStreamPlugin
from common.packet import FramePacket
from core.tracker.ball_tracker.tracknet_wrapper import TrackNetWrapper
from core.tracker.human_tracker.byte_wrapper import ByteTrackWrapper
from core.detectors.yolo_pose import YOLOPoseDetector
from core.inference.bst_processor import BSTFeatureGenerator
from core.classifiers.bst_classifier import BSTClassifier
from modules.event_detector import HitTrigger
from modules.spatial_calc import CourtMapper, KinematicsAnalyzer
from utils.logger import logger, PerformanceTimer, catch_errors
from utils.coords import scale_coords
from utils.writer import ResultWriter
from utils.config_loader import cfg

class StateCache:
    """维护历史数据，为 BST 等时序模型提供窗口数据"""
    def __init__(self, max_len=30):
        self.max_len = max_len
        self.player_states = {} # track_id -> deque of skeletons
        self.ball_states = deque(maxlen=max_len)
        self.ball_trail = deque(maxlen=100) # 专门用于 UI 渲染的更长轨迹
    
    def update(self, packet: FramePacket):
        # 更新球坐标历史
        self.ball_states.append(packet.ball_coord)
        self.ball_trail.append(packet.ball_coord)
        
        # 更新球员历史
        active_ids = []
        for skel in packet.skeletons:
            tid = skel.get("player_id")
            if tid is not None:
                if tid not in self.player_states:
                    self.player_states[tid] = deque(maxlen=self.max_len)
                self.player_states[tid].append(skel)
                active_ids.append(tid)

class InferenceEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InferenceEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.stream_plugin: Optional[IStreamPlugin] = None
        
        # --- 使用配置加载器初始化模型 ---
        self.yolo_detector = YOLOPoseDetector(
            model_path=cfg.get("YOLO.weight_path"),
            device=cfg.get("Hardware.device_id"),
            use_trt=cfg.get("YOLO.use_tensorrt")
        )
        self.ball_tracker = TrackNetWrapper(
            model_path=cfg.get("TrackNetV3.weight_path")
        )
        self.human_tracker = ByteTrackWrapper(
            track_thresh=cfg.get("ByteTrack.track_thresh"),
            match_thresh=cfg.get("ByteTrack.match_thresh")
        )
        self.bst_generator = BSTFeatureGenerator()
        self.bst_classifier = BSTClassifier(
            model_path=cfg.get("BST.weight_path"),
            seq_len=cfg.get("BST.window_size", 31),
            device=cfg.get("Hardware.device_id")
        )
        self.hit_trigger = HitTrigger(
            angle_threshold=cfg.get("Pipeline.hit_angle", 90.0),
            dist_threshold_meters=cfg.get("Pipeline.hit_dist", 1.5)
        )
        self.court_mapper = CourtMapper()
        self.kinematics = KinematicsAnalyzer()
        self.state_cache = StateCache(max_len=cfg.get("BST.window_size", 100))
        self._pre_rally_candidates = deque(maxlen=6)
        
        # --- 线程通信 ---
        max_q = cfg.get("Stream.max_queue_size", 30)
        self.input_queue = queue.Queue(maxsize=max_q)
        self.output_queue = queue.Queue(maxsize=max_q)
        
        # --- 结果持久化 ---
        out_dir = cfg.get("Logging.output_dir", "data/output")
        self.result_writer = ResultWriter(f"{out_dir}/results.json", format="json")
        
        self.running = False
        self.paused = False # 播放暂停状态
        self.hit_count = 0  # 累计击球数
        self.is_rally = False # 当前是否处于回合中
        self._threads = []
        self._initialized = True

    def set_stream_source(self, plugin: IStreamPlugin):
        """注入流插件"""
        self.stream_plugin = plugin

    def has_source(self):
        """检查是否有已连接的流源"""
        return self.stream_plugin is not None and self.stream_plugin.is_opened

    def load_source(self, config: dict) -> bool:
        """根据配置加载流源，并重置缓存"""
        try:
            from plugins.streams.file_stream import FileStreamPlugin
            from plugins.streams.rtsp_stream import RTSPStreamPlugin
            
            # 清理历史缓存防止跨视频干扰
            if hasattr(self.ball_tracker, 'frame_buffer'):
                self.ball_tracker.frame_buffer.clear()
            self.state_cache.player_states.clear()
            self.state_cache.ball_states.clear()
            self._pre_rally_candidates.clear()
            if self.bst_generator:
                self.bst_generator.player_buffers.clear()
            if self.bst_classifier:
                self.bst_classifier.history.clear()
            self.hit_count = 0 
            self.is_rally = False
            
            stype = config.get("type")
            path = config.get("path") or config.get("url") or config.get("index")
            
            if stype == "file":
                self.stream_plugin = FileStreamPlugin()
            elif stype in ["rtsp", "camera"]:
                self.stream_plugin = RTSPStreamPlugin()
            else:
                return False
                
            return self.stream_plugin.connect(str(path))
        except Exception as e:
            logger.error(f"Failed to load source: {e}")
            return False

    def close_source(self):
        """关闭当前的流源"""
        if self.stream_plugin:
            self.stream_plugin.release()
            self.stream_plugin = None

    def process_frame(self, packet: FramePacket) -> FramePacket:
        """同步处理单帧：用于 UI 线程直接调用或作为测试"""
        with PerformanceTimer("EndToEnd_Latency", packet.frame_id):
            orig_h, orig_w = packet.image.shape[:2]
            
            # --- [Debug] 核心环节日志输出 ---
            logger.debug(f"Frame {packet.frame_id}: Process Start")
            
            # 0. 更新空间映射 (如果 UI 传递了角点)
            if packet.court_info and "corners" in packet.court_info:
                self.court_mapper.update_homography(packet.court_info["corners"])
            
            # 1. YOLO-Pose: 获取 Bbox, Keypoints
            if self.yolo_detector:
                all_players, _ = self.yolo_detector.predict(packet.image)
                # 保存全量的检测结果供后续的球过滤（尤其是裁判等非玩家目标）
                packet.metadata['all_detected_persons'] = all_players
                
                logger.debug(f"YOLO: Detected {len(all_players)} potential objects")
                # 增强过滤逻辑：排除杂色及场外干扰（如裁判）
                scored_players = []
                for p in all_players:
                    x1, y1, x2, y2, conf = p['bbox']
                    if (x2 - x1) < 20 or (y2 - y1) < 40: continue # 过滤过小目标
                    
                    # 空间优先级得分：如果在标定的球场范围内，获得 0.5 的额外加分
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    spatial_bonus = 0
                    if self.court_mapper.last_valid_matrix is not None:
                        real = self.court_mapper.pixel_to_real(cx, cy)
                        if real and (-1 < real[0] < 7.1 and -1 < real[1] < 14.4): # 考虑边界容差
                            spatial_bonus = 0.5
                    scored_players.append((p, conf + spatial_bonus))
                
                # 严格保留前 2 名得分最高的目标
                top_scored = sorted(scored_players, key=lambda x: x[1], reverse=True)[:2]
                packet.skeletons = [s[0] for s in top_scored]
                logger.debug(f"Filter: Retained {len(packet.skeletons)} players")

            # 2. ByteTrack: 更新 track_id
            if self.human_tracker and packet.skeletons:
                boxes = [s['bbox'] for s in packet.skeletons if 'bbox' in s]
                if boxes:
                    tracks = self.human_tracker.update(np.array(boxes), packet.image.shape[:2])
                    # 匹配 ID
                    for skel in packet.skeletons:
                        sb = skel['bbox']
                        for t in tracks:
                            tb = t.tlbr
                            if abs(sb[0]-tb[0]) < 20 and abs(sb[1]-tb[1]) < 20: # 扩大容差
                                skel["player_id"] = t.track_id
                                break
                    
                for skel in packet.skeletons:
                    if "player_id" not in skel: skel["player_id"] = -1

            # 3. TrackNetV3: 预测球坐标
            if self.ball_tracker:
                ball_res = self.ball_tracker.predict(packet.image)
                if ball_res:
                    bx, by, b_conf = ball_res[0] * orig_w, ball_res[1] * orig_h, ball_res[2]
                    
                    # --- 增强过滤逻辑：解决水壶盖/杂色干扰 ---
                    is_valid_ball = True
                    
                    # (1) 空间过滤：如果在标定范围外太远，直接舍弃
                    if self.court_mapper.last_valid_matrix is not None:
                        real = self.court_mapper.pixel_to_real(bx, by)
                        if real:
                            # 正常羽毛球场宽 6.1m, 长 13.4m。考虑界外救球，容差设定为左右 2m，上下 3m
                            if not (-2.0 < real[0] < 8.1 and -2.0 < real[1] < 15.4):
                                is_valid_ball = False
                                logger.debug(f"Ball Filter: Out of bounds at {real}")

                    # (2) 人体/静态遮罩过滤：球点落在裁判框或处于静止状态需排除
                    all_persons = packet.metadata.get('all_detected_persons', [])
                    for p in all_persons:
                        x1, y1, x2, y2, p_conf = p['bbox']
                        # 扩大一点范围，覆盖水壶等周围杂物
                        if (x1 - 10) < bx < (x2 + 10) and (y1 - 10) < by < (y2 + 10):
                            # 如果该目标在场外（可能是坐着的裁判/观众），直接判定无效
                            cx_p, cy_p = (x1 + x2) / 2, (y1 + y2) / 2
                            if self.court_mapper.last_valid_matrix is not None:
                                p_real = self.court_mapper.pixel_to_real(cx_p, cy_p)
                                if p_real and (p_real[0] < -0.5 or p_real[0] > 6.6 or p_real[1] < -0.5 or p_real[1] > 13.9):
                                    is_valid_ball = False
                                    logger.debug("Ball Filter: Masked by person outside court")
                                    break

                    # (3) 动态过滤：羽毛球必须在运动
                    if is_valid_ball and len(self.state_cache.ball_states) > 5:
                        recent_balls = [b for b in list(self.state_cache.ball_states)[-10:] if b is not None]
                        if len(recent_balls) >= 5:
                            # 计算最近几帧的位移标准差，如果极小（静止），则是假正例
                            dists = [((bx - rb[0])**2 + (by - rb[1])**2)**0.5 for rb in recent_balls]
                            # 如果连续 5+ 帧位移都小于 3 像素，判定为静止干扰物
                            if all(d < 3 for d in dists):
                                is_valid_ball = False
                                logger.debug("Ball Filter: Static object detected (water bottle cap etc.)")

                    # (4) 突变过滤：羽毛球不可能瞬间从一个位置跳到屏幕另一端
                    if is_valid_ball and len(self.state_cache.ball_states) > 0:
                        last_valid = next((b for b in reversed(self.state_cache.ball_states) if b is not None), None)
                        if last_valid:
                            pixel_dist = ((bx - last_valid[0])**2 + (by - last_valid[1])**2)**0.5
                            # 如果单帧跳变超过画面对角线的 1/4，且没有伴随高速趋势，判定为误检跳跃
                            if pixel_dist > (orig_w**2 + orig_h**2)**0.5 / 4:
                                is_valid_ball = False
                                logger.debug(f"Ball Filter: Speed jump detected ({pixel_dist:.1f}px)")

                    # (5) 速度一致性过滤：当前点与线性预测偏差过大时视为乱飞
                    if is_valid_ball and len(self.state_cache.ball_states) >= 2:
                        last_ball = self.state_cache.ball_states[-1]
                        prev_ball = self.state_cache.ball_states[-2]
                        if last_ball and prev_ball:
                            vx, vy = last_ball[0] - prev_ball[0], last_ball[1] - prev_ball[1]
                            pred_x, pred_y = last_ball[0] + vx, last_ball[1] + vy
                            dev = ((bx - pred_x)**2 + (by - pred_y)**2)**0.5
                            diag = (orig_w**2 + orig_h**2)**0.5
                            dev_thresh = max(120.0, diag * 0.08)
                            if dev > dev_thresh and b_conf < 0.7:
                                is_valid_ball = False
                                logger.debug(f"Ball Filter: Velocity outlier ({dev:.1f}px)")

                    if is_valid_ball and not self.is_rally:
                        # 预热门槛：开赛前必须出现明显位移，避免锁定静态小物体
                        self._pre_rally_candidates.append((bx, by))
                        if len(self._pre_rally_candidates) < 3:
                            is_valid_ball = False
                            logger.debug("Ball Filter: Pre-rally warmup (insufficient motion history)")
                        else:
                            first = self._pre_rally_candidates[0]
                            last = self._pre_rally_candidates[-1]
                            motion = ((last[0] - first[0])**2 + (last[1] - first[1])**2)**0.5
                            if motion < 12:
                                is_valid_ball = False
                                logger.debug("Ball Filter: Pre-rally static lock ignored")

                    if is_valid_ball:
                        packet.ball_coord = (bx, by, b_conf)
                        logger.debug(f"Ball: Found at ({bx:.1f}, {by:.1f}) conf={b_conf:.2f}")
                        if not self.is_rally:
                            self._pre_rally_candidates.clear()
                
                # 平滑逻辑：如果当前帧丢球（或被过滤），但历史中有球且处于高速运动，进行线性预测插值
                if packet.ball_coord is None and len(self.state_cache.ball_states) >= 2:
                    last_ball = self.state_cache.ball_states[-1]
                    prev_ball = self.state_cache.ball_states[-2]
                    if last_ball and prev_ball:
                        vx, vy = last_ball[0] - prev_ball[0], last_ball[1] - prev_ball[1]
                        if abs(vx) > 5 or abs(vy) > 5:
                            packet.ball_coord = (last_ball[0] + vx, last_ball[1] + vy, 0.4) 
                            packet.metadata['is_interpolated'] = True
                            logger.debug("Ball: Interpolated position used")

            # 4. 击球动作与回合检测 (基于空间投影和运动学)
            if self.ball_coord_valid(packet.ball_coord):
                # 简单回合逻辑：如果球在运动且置信度高，认为回合进行中
                self.is_rally = True
                
                # 检查是否发生击球 (基于物理轨迹剧烈转向)
                is_hit = self.hit_trigger.check_hit(packet, self.court_mapper)
                if is_hit:
                    packet.metadata['is_hit'] = True
                    logger.info(f"Event: HIT DETECTED at frame {packet.frame_id}")
                    # 距离最近的球员即为击球者
                    hitter_id = self._assign_hitter(packet)
                    if hitter_id != -1:
                        packet.metadata['hitter_id'] = hitter_id
                    # 击球计数不依赖 hitter_id，避免统计缺失
                    self.hit_count += 1
                    packet.metadata['hit_count'] = self.hit_count
            else:
                # 如果连续多帧没球，认为回合结束
                if not hasattr(self, '_no_ball_frames'): self._no_ball_frames = 0
                self._no_ball_frames += 1
                if self._no_ball_frames > 15:
                    if self.is_rally: logger.info("Event: Rally Ended")
                    self.is_rally = False
                    self._no_ball_frames = 0
                    self._pre_rally_candidates.clear()

            packet.metadata['is_rally'] = self.is_rally
            packet.metadata['hit_count'] = self.hit_count

            # 5. BST Feature Generation & Classification
            if self.bst_generator:
                self.bst_generator.update(packet)
                
            for skel in packet.skeletons:
                track_id = skel.get("player_id", -1)
                if track_id == -1:
                    continue

                # 始终维护 BST 历史序列，避免只在候选帧更新导致序列不足
                self.bst_classifier.update_history(track_id, skel["keypoints"][:, :2], packet.ball_coord)

                # 命中候选或明确击球时才触发分类
                if ("bst_input" in skel) or packet.metadata.get("is_hit"):
                    action = self.bst_classifier.predict(track_id)
                    if action:
                        logger.info(f"BST: Player {track_id} Action -> {action}")
                        skel["action"] = action
                        packet.stroke_action = {"label": action, "player_id": track_id, "conf": 0.9}
                        packet.metadata['is_new_event'] = True

            # 6. StateCache 更新
            self.state_cache.update(packet)
            # 将球的历史轨迹传递给 Packet，使用专门的 ball_trail (100帧) 以获得更明显的拖尾效果
            packet.metadata['ball_history'] = list(self.state_cache.ball_trail)
            
        return packet

    def ball_coord_valid(self, coord):
        return coord is not None and len(coord) >= 3 and coord[2] > 0.3

    def _assign_hitter(self, packet) -> int:
        """寻找距离球最近的球员作为击球者"""
        if not packet.ball_coord or not packet.skeletons:
            return -1
        
        bx, by = packet.ball_coord[:2]
        min_dist = float('inf')
        hitter_id = -1
        
        for skel in packet.skeletons:
            bbox = skel.get("bbox")
            if bbox is not None:
                cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                dist = (bx-cx)**2 + (by-cy)**2
                if dist < min_dist:
                    min_dist = dist
                    hitter_id = skel.get("player_id", -1)
        
        return hitter_id

    def get_next_packet(self) -> Optional[FramePacket]:
        """获取并同步读取下一帧数据"""
        if not self.has_source():
            return None
        
        # 处理暂停逻辑：如果暂停，则保持循环直到继续
        while self.paused and self.running:
            time.sleep(0.1)
            
        packet = self.stream_plugin.read()
        if packet:
            # 执行推理处理
            try:
                self.process_frame(packet)
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            packet.metadata['infer_fps'] = self.calculate_fps()
            packet.metadata['stream_fps'] = self.stream_plugin.cap.get(cv2.CAP_PROP_FPS) if hasattr(self.stream_plugin, 'cap') else 30.0
            
        return packet

    def calculate_fps(self):
        # 简单 FPS 计算
        if not hasattr(self, '_last_time'):
            self._last_time = time.time()
            self._frame_count = 0
            return 0.0
        
        self._frame_count += 1
        now = time.time()
        duration = now - self._last_time
        
        if duration >= 1.0:
            self._current_fps = self._frame_count / duration
            self._last_time = now
            self._frame_count = 0
            return self._current_fps
            
        return getattr(self, '_current_fps', 0.0)

    def start(self):
        """启动流水线"""
        if not self.stream_plugin:
            logger.error("No stream plugin set.")
            return

        self.running = True
        self.result_writer.start()
        
        # 线程 1: 采集
        t1 = threading.Thread(target=self._ingest_loop, name="IngestThread", daemon=True)
        # 线程 2: 处理
        t2 = threading.Thread(target=self._process_loop, name="ProcessThread", daemon=True)
        
        self._threads = [t1, t2]
        for t in self._threads:
            t.start()
        
        logger.info("InferenceEngine initialized and started.")

    def stop(self):
        self.running = False
        self.result_writer.stop()
        for t in self._threads:
            t.join(timeout=1.0)
        if self.stream_plugin:
            self.stream_plugin.release()
        logger.info("Engine shutdown complete.")

    def _ingest_loop(self):
        """读取线程：将 FramePacket 放入队列"""
        while self.running and self.stream_plugin.is_opened:
            packet = self.stream_plugin.read()
            if packet:
                # 性能控制：如果堆积超过 10 帧，清空旧帧，确保实时性
                q_size = self.input_queue.qsize()
                if q_size > 10:
                    logger.warning(f"Pipeline lagging! Queue size: {q_size}. Dropping stale frames.")
                    while not self.input_queue.empty():
                        try:
                            self.input_queue.get_nowait()
                        except queue.Empty:
                            break
                
                self.input_queue.put(packet)

    @catch_errors
    def _process_loop(self):
        """主处理流水线：检测 -> 追踪 -> 预测 -> 缓存"""
        while self.running:
            try:
                packet: FramePacket = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with logger.contextualize(frame_id=packet.frame_id):
                self.process_frame(packet)
                
                # 6. Result Persistence: 异步写入结果
                self.result_writer.write_packet(packet)

                # 输出处理结果
                if self.output_queue.full():
                    self.output_queue.get_nowait()
                self.output_queue.put(packet)
                
                if packet.frame_id % 30 == 0:
                    latency = PerformanceTimer.get_average("EndToEnd_Latency")
                    logger.info(f"Pipeline Snapshot | F:{packet.frame_id} | Avg Latency: {latency:.2f}ms | Q_In: {self.input_queue.qsize()}")

    def get_latest_packet(self) -> Optional[FramePacket]:
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
