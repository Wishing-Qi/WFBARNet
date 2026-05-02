# 当前羽毛球预测全流程

本文描述当前 PyQt6 桌面端的羽毛球分析链路，覆盖视频/摄像头输入、球轨迹预测、人体姿态、球场投影、击球点识别、BST 击球动作分类、位移统计和调试日志。

对应主要入口：

- `apps/pyqt6/main.py`
- `apps/pyqt6/controllers/analysis_controller_runtime.py`
- `src/models/track_branch.py`
- `src/models/pose_branch.py`
- `src/postprocess/track_filter.py`
- `src/utils/visualize.py`

## 总览

```text
视频文件/摄像头
  -> 读取 prev/current/next 三帧
  -> OpenCV 球场线服务异步更新 court_prediction
  -> TrackNet 预测羽毛球热力图候选点
  -> YOLO26s-Pose 周期性预测人体姿态
  -> CourtPoseTargetTracker 稳定上/下半场球员
  -> BallTrackFilter 融合候选点、球场、人体框、运动学约束
  -> FrameResult(frame_id, pose, track)
  -> TrackTrailRenderer 绘制轨迹并识别击球点 hit_event
  -> BSTStrokeRecognizer 在 hit_event 触发时分类击球动作
  -> PlayerDistanceAccumulator 统计双方位移
  -> UI 显示 + CSV/JSONL 日志
```

## 1. 应用启动

PyQt6 入口是 `apps/pyqt6/main.py`：

1. 创建 `QApplication` 和 `MainWindow`。
2. 创建 `CourtDetectionService`。
3. 创建 `MainController(window, court_service=...)`。
4. 控制器加载模型、绑定按钮事件、启动分析 worker。

主要模型默认路径：

- 姿态模型：`assets/weights/pose/yolo26s-pose.pt`
- 轨迹模型：`assets/weights/track/model_best.pt` 或 `.engine`
- BST 模型：`assets/weights/bst/bst_CG_AP_JnB_bone_merged_10.pt`

## 2. 输入模式

当前有两条运行路径：

- 视频文件：`TrackNetPlaybackWorker`
- 摄像头实时流：`CameraInferenceWorker`

两条路径的核心推理逻辑基本一致，差别主要在帧读取和时间戳：

- 视频模式使用 `cv2.VideoCapture` 读取文件，并使用视频 FPS 和 `CAP_PROP_POS_MSEC` 计算时间。
- 摄像头模式持续读取实时帧，时间戳来自 `QElapsedTimer`。

TrackNet 使用三帧窗口：

```text
[prev_frame, current_frame, next_frame]
```

这样模型能利用短时间运动信息预测当前帧羽毛球位置。

## 3. 球场线预测

球场线由 `apps/pyqt6/services/court_detection_service.py` 中的异步 worker 维护。

每帧分析时，worker 会：

1. 把当前帧提交给球场线服务。
2. 读取最近一次有效 `court_prediction`。
3. 将 `court_prediction` 传给姿态、轨迹滤波和位移统计模块。

`court_prediction` 主要提供：

- `valid`
- `image_to_court_h`
- `corners`
- `projected_lines`

其中 `image_to_court_h` 用于把图像坐标投影到真实球场平面，球场尺寸按厘米计：

- 宽度：`610 cm`
- 长度：`1340 cm`

## 4. TrackNet 羽毛球候选点

轨迹模型由 `src/models/track_branch.py` 封装为 `TrackBranch`。

当前 PyQt6 创建参数：

```text
input_size = (512, 288)
score_thr = 0.35
backend = pytorch 或 tensorrt
```

如果轨迹权重后缀是 `.engine`，则使用 TensorRT 后端；否则使用 PyTorch `TrackNetV3`。

每帧调用：

```python
infer_candidate_results([prev_frame, current_frame, next_frame])
```

流程：

1. `preprocess_track_window` 将三帧缩放到 TrackNet 输入尺寸。
2. TrackNet 输出羽毛球热力图。
3. `decode_track_heatmap_candidates` 对热力图做阈值二值化。
4. 使用 `connectedComponentsWithStats` 找连通区域。
5. 每个区域按峰值、均值、面积和紧致度排序。
6. 返回多个 `TrackResult` 候选点。

`TrackResult` 数据结构：

```python
TrackResult(
    ball_xy=[x, y],
    visible=0 or 1,
    score=float,
    heatmap_shape=[h, w],
)
```

## 5. BallTrackFilter 轨迹滤波

`src/postprocess/track_filter.py` 中的 `BallTrackFilter` 是当前轨迹稳定性的核心。

输入：

- TrackNet 候选点列表
- 当前帧尺寸
- 球场预测 `court_prediction`
- 当前姿态人体框 `person_bboxes`

输出：

- 单个稳定后的 `TrackResult`

主要规则：

1. 球场区域过滤  
   候选点需要落在球场投影区域或扩展空中区域内。

2. 静态热点过滤  
   INT8 量化模型容易在边缘或固定区域产生稳定假点，滤波器会记录并抑制这些热点。

3. 运动预测和门控  
   使用上一帧位置、速度、加速度、横向误差、反向运动距离等判断候选点是否可信。

4. 抛物线预测  
   最近真实点足够多时拟合轻量二次曲线，用于短时漏检补点。

5. 人体遮挡处理  
   如果预测轨迹穿过人体框，人体框内候选点会被降权或拒绝，同时允许更长时间的短时预测续航。

6. 顶部出画处理  
   球向画面顶部快速飞出时，进入短时 suppress，避免顶部边缘假点被误认为真实轨迹。

7. 远点重锁保护  
   当前版本限制 `impact_direction_change` 重锁必须接近预测位置，避免从中场突然跳到远处高分假点。

8. 顶部边缘假点保护  
   如果当前真实轨迹还远离顶部，但突然出现顶部边缘候选点，会按 `top_edge_hallucination` 拒绝。

调试时重点看 `track_debug.csv` 中这些字段：

- `action`
- `reason`
- `input_x/input_y/input_score`
- `output_x/output_y/output_score`
- `pred_x/pred_y`
- `candidates`

常见 `action/reason`：

- `accept/passes_motion_gate`
- `coast/parabola_prediction`
- `coast/person_occlusion_prediction`
- `reject/top_edge_hallucination`
- `reject/candidate_failed_motion_gate`
- `top_exit_enter/likely_top_exit`
- `drop_lock/max_missed_frames`

## 6. 人体姿态预测

姿态模型由 `src/models/pose_branch.py` 封装为 `PoseBranch`。

当前 PyQt6 参数：

```text
backend = yolo26s-pose
conf_thr = 0.35
max_persons = 12
yolo_imgsz = 960
yolo_crop_pose = True
yolo_crop_imgsz = 640
yolo_crop_padding = 0.30
yolo_crop_min_box_conf = 0.45
yolo_court_filter = True
yolo_court_required = True
```

姿态不是每帧都跑，当前 stride：

```text
POSE_INFERENCE_STRIDE = 2
```

也就是每 2 帧推理一次，其他帧由跟踪器维持。

输出结构：

```python
PersonPoseResult(
    person_id=int,
    bbox=[x1, y1, x2, y2],
    keypoints=[[x, y], ...],
    scores=[score, ...],
    person_score=float,
)
```

## 7. 姿态目标稳定

`CourtPoseTargetTracker` 负责从多个人体检测中稳定保留双方球员。

逻辑：

1. 使用脚踝点或 bbox 底部中心作为人体落地点。
2. 通过 `image_to_court_h` 投影到球场平面。
3. 按球场上下半场分组。
4. 每个半场最多保留一个目标。
5. 对 bbox 和关键点做平滑。
6. 短时漏检时按速度预测人体位置。

这一步的结果会传给：

- 视频画面姿态绘制
- `BallTrackFilter` 人体遮挡过滤
- 击球点姿态辅助
- 球员位移统计

## 8. 球员位移统计

`src/postprocess/player_distance.py` 中的 `PlayerDistanceAccumulator` 负责统计双方位移。

流程：

1. 从姿态中取脚踝点；脚踝不可用时用 bbox 底部中心。
2. 使用 `image_to_court_h` 投影到球场平面。
3. 按 `person_id` 或 fallback index 映射到上方/下方球员。
4. 累加相邻投影点距离。

过滤规则：

- 小于 `2 cm` 的抖动不计入。
- 大于 `180 cm` 的单步跳变视为跟踪断裂，不计入。

UI 显示单位为米：

```text
上方球员: xx.xx 米
下方球员: xx.xx 米
```

## 9. FrameResult

每帧统一封装为：

```python
FrameResult(
    frame_id=frame_id,
    pose=last_pose,
    track=track,
)
```

它是后续可视化、击球点检测、BST 输入、日志导出的统一数据单元。

## 10. 击球点识别

击球点由 `src/utils/visualize.py` 中的 `TrackTrailRenderer` 识别。

每帧处理：

1. 更新人体手腕运动状态。
2. 如果球可见，把球点加入短历史队列。
3. 使用连续三点判断是否发生速度突变或方向突变。
4. 结合人体姿态给出 pose assist 分数。
5. 过滤顶部出画、人体遮挡假点、地板弹跳假点。
6. 如果通过判断，生成 `hit_event`。

`hit_event` 格式：

```json
{
  "frame_id": 123,
  "timestamp_ms": 2050,
  "ball_xy": [x, y]
}
```

关键判断因素：

- 前后速度
- 转向角
- 速度变化比例
- 与手腕距离
- 手腕速度
- 手臂伸展程度
- 是否在人体框内
- 是否像地板弹跳
- 是否接近顶部出画区域

当前版本提高了腕部姿态辅助的覆盖范围，并允许中等姿态置信度覆盖部分弹跳/遮挡过滤，以减少漏检。

## 11. BST 击球动作分类

`src/models/bst_stroke_runtime.py` 中的 `BSTStrokeRecognizer` 在出现 `hit_event` 时触发。

流程：

1. 持续维护最近 `seq_len` 帧 `FrameResult` 缓冲区。
2. `hit_event` 触发时，从缓冲区构建 BST 输入。
3. 使用人体姿态、羽毛球轨迹、球场位置特征组成 batch。
4. 调用 BST 模型做 25 类击球动作分类。
5. 输出 top5 和置信度。

输出会显示到 UI 动作列表，并写入日志：

```text
[BST] hit 12.34s -> 某类击球 (xx.x%)
```

## 12. UI 显示

每帧可视化由 `TrackTrailRenderer.draw_on` 完成：

- 绘制人体骨架和 bbox。
- 绘制当前球点。
- 绘制短时轨迹尾迹。
- 绘制击球点红色 marker。

为了减少卡顿，UI 不一定每帧刷新；但即使跳过显示帧，也会调用：

```python
trail_renderer.update_hit_detection(...)
```

因此击球点检测不会因为 UI 降帧而停止。

## 13. 日志输出

设置页中打开 `Debug Logs / Write frame analysis logs` 后，会输出两类文件到：

```text
outputs/pyqt_debug/
```

### TrackDebug CSV

文件名：

```text
*_track_debug.csv
```

记录轨迹滤波器每帧的输入、输出、预测位置、决策原因和候选点列表。

适合排查：

- 假点
- 漏检
- 重锁错误
- 顶部出画
- 人体遮挡
- INT8 热点漂移

### FrameLog JSONL

文件名：

```text
*_frame_log.jsonl
```

每行一个 JSON 对象，包含：

```json
{
  "frame_id": 123,
  "timestamp_ms": 2050,
  "ball": {
    "xy": [x, y],
    "visible": 1,
    "score": 0.72
  },
  "pose": [
    {
      "person_id": 0,
      "bbox": [x1, y1, x2, y2],
      "person_score": 0.91,
      "keypoints": [[x, y]],
      "keypoint_scores": [0.9]
    }
  ],
  "hit_event": null
}
```

如果某帧触发击球点，`hit_event` 会写入击球帧、时间戳和球坐标。

## 14. 当前主要调参入口

### 球轨迹

文件：`src/postprocess/track_filter.py`

常用参数：

- `min_confidence`
- `soft_min_confidence`
- `base_gate_px`
- `max_gate_px`
- `max_speed_px_per_sec`
- `parabola_*`
- `person_occlusion_*`
- `top_exit_*`
- `top_edge_hallucination_*`
- `impact_relock_*`
- `static_hotspot_*`

### 击球点

文件：`src/utils/visualize.py`

常用参数：

- `hit_min_speed_px_per_sec`
- `hit_min_turn_deg`
- `hit_min_speed_change_ratio`
- `hit_cooldown_seconds`
- `hit_pose_assist_*`
- `hit_floor_bounce_*`

### 姿态

文件：`apps/pyqt6/controllers/analysis_controller_runtime.py`

常用参数：

- `POSE_INFERENCE_STRIDE`
- `POSE_CANDIDATE_LIMIT`
- `POSE_YOLO_IMGSZ`
- `POSE_CROP_IMGSZ`
- `POSE_CROP_PADDING`
- `POSE_COURT_MARGIN_CM`

## 15. 常见问题定位

### 球轨迹乱飘

优先看 `*_track_debug.csv`：

1. 找 `action=relock_accept` 或 `reason=impact_direction_change`。
2. 看 `input_x/input_y` 和 `pred_x/pred_y` 的距离是否过大。
3. 看 `candidates` 里是否存在高分但远离预测轨迹的候选点。
4. 如果发生在画面顶部，重点看 `top_exit_*` 和 `top_edge_hallucination_*`。

### 球经过人体后断轨

检查：

- `frame_log.jsonl` 中对应帧是否有人体 bbox。
- `track_debug.csv` 中是否出现 `coast/person_occlusion_prediction`。
- 如果没有，说明姿态未覆盖遮挡区域或姿态 stride 导致当前帧没有可用人体框。

### 击球点漏检

检查 `frame_log.jsonl`：

1. 找球轨迹发生明显转向的三帧。
2. 看中间帧附近是否有手腕 keypoint。
3. 看 `hit_event` 是否为 null。
4. 如果腕部距离较近但未识别，调 `hit_pose_assist_*`。
5. 如果落地弹跳误检，调 `hit_floor_bounce_*`。

### BST 没有输出

BST 只有在 `hit_event` 非空时才会触发。

检查：

- 是否加载了 BST 权重。
- `frame_log.jsonl` 是否有 `hit_event`。
- UI 日志中是否有 `[BST] stroke inference disabled after error`。

## 16. CLI runner

除 PyQt6 外，项目还有 CLI runners：

- `src/runners/track_video_runner.py`
- `src/runners/pose_video_runner.py`
- `src/runners/unified_runner.py`
- `src/runners/tracknet_realtime_runner.py`

这些 runner 也会复用 `TrackBranch`、`PoseBranch`、`BallTrackFilter`、`FrameResult` 和 exporter，但当前主要交互体验与新增日志功能集中在 PyQt6 runtime。
