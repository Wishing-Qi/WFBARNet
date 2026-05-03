# 当前轨迹推理流程

更新时间：2026-05-03

本文档记录当前 PyQt6 应用中的羽毛球轨迹推理、轨迹滤波、击球点检测、姿态辅助和调试日志流程。文档描述的是当前代码实现，不是理想设计。

主要相关文件：

- `apps/pyqt6/controllers/analysis_controller_runtime.py`
- `src/models/track_branch.py`
- `src/preprocess/track.py`
- `src/postprocess/track.py`
- `src/postprocess/track_filter.py`
- `src/postprocess/pose.py`
- `src/postprocess/player_distance.py`
- `src/utils/visualize.py`
- `src/utils/exporters.py`

## 1. 总体流程

```text
视频/摄像头帧
  -> 组织 TrackNet 三帧窗口 [prev, current, next]
  -> 场地检测服务输出 court_prediction
  -> TrackBranch 输出候选羽毛球点列表
  -> PoseBranch 输出人体姿态候选
  -> CourtPoseTargetTracker 选择并稳定双方球员姿态
  -> BallTrackFilter 过滤候选点、预测丢失帧、relock 新轨迹
  -> FrameResult(frame_id, pose, track)
  -> TrackTrailRenderer 更新轨迹尾迹和 hit_event
  -> BSTStrokeRecognizer 根据 hit_event 做击球动作分类
  -> PlayerDistanceAccumulator 统计球员场地投影位移
  -> UI 显示，并可写入 track_debug.csv / frame_log.jsonl
```

视频回放和摄像头实时流使用相同的核心组件。差异主要在时间戳来源：

- 视频回放使用 `cv2.VideoCapture` 的帧时间和视频 FPS。
- 摄像头实时流使用 `QElapsedTimer` 作为当前时间。

无论当前帧是否实际渲染到 UI，`TrackTrailRenderer.update_hit_detection(...)` 都会被调用，因此 `hit_event` 和 JSONL 日志不会因为降低显示帧率而中断。

## 2. 运行时主循环

运行时在 `analysis_controller_runtime.py` 中创建以下状态对象：

- `BallTrackFilter(fps=fps, debug_enabled=...)`
- `CourtPoseTargetTracker(...)`
- `PlayerDistanceAccumulator()`
- `TrackTrailRenderer(fps=fps, history_seconds=0.5)`
- 可选的 `BSTStrokeRecognizer`

每一帧的核心步骤：

1. 计算当前 `frame_id` 和 `timestamp_ms`。
2. 将当前帧提交给场地检测服务，读取最新 `court_prediction`。
3. 用 `[prev_frame, current_frame, next_frame]` 调用 `TrackBranch.infer_candidate_results(...)`。
4. 每 `POSE_INFERENCE_STRIDE = 2` 帧运行一次姿态推理。
5. 用 `CourtPoseTargetTracker` 更新稳定后的球员姿态。
6. 用 `BallTrackFilter.update_candidates(...)` 从候选球点中输出最终 `TrackResult`。
7. 组装 `FrameResult(frame_id, pose, track)`。
8. 更新轨迹尾迹和击球事件。
9. 写入可选调试日志。
10. 更新 UI payload，包括球点、姿态、场地、球员位移、击球分类等。

当 TrackNet 使用 TensorRT 后端且姿态推理刚好到期时，轨迹和姿态会通过 `ThreadPoolExecutor(max_workers=2)` 并行推理。

## 3. TrackBranch 推理

`TrackBranch` 支持两种后端：

- `.engine`：TensorRT 后端，通常用于 INT8 量化版本。
- 其他权重文件：PyTorch `TrackNetV3`。

当前 PyQt6 默认创建参数：

```text
input_size = (512, 288)
score_thr = 0.35
max_candidates = 5
candidate_score_thr_ratio = 0.6
```

因此候选点解码阈值为：

```text
candidate_score_thr = score_thr * candidate_score_thr_ratio = 0.21
```

注意：候选列表可以包含 `score < 0.35` 的弱候选，但 `BallTrackFilter` 的正式 measurement 仍默认要求 `min_confidence = 0.35`。

## 4. TrackNet 输入预处理

`preprocess_track_window(...)` 要求刚好 3 帧：

```text
[prev_frame, current_frame, next_frame]
```

预处理步骤：

1. 将三帧都 resize 到 `512 x 288`。
2. BGR 转 RGB。
3. 三帧按通道拼接为 `9 x 288 x 512`。
4. 转为 float32 tensor，并除以 `255.0`。
5. 记录缩放比例 `scale_x`、`scale_y`，用于把热力图坐标还原到原始视频尺寸。

TrackNet 输出取第 1 个前景平面作为羽毛球热力图。

## 5. 热力图候选点解码

候选点解码在 `src/postprocess/track.py` 中完成。

流程：

1. 对热力图按阈值生成二值 mask。
2. 使用 `cv2.connectedComponentsWithStats(mask, 8)` 找连通域。
3. 每个连通域计算：
   - `peak`：区域内最大热力图分数。
   - `mean`：区域平均分数。
   - `area`：连通域面积。
   - `compactness`：面积 / 外接矩形面积。
4. 使用热力图值加权计算连通域中心。
5. 按 `(peak, mean, min(area, 24), compactness)` 排序。
6. 最多保留 `max_candidates = 5` 个候选。
7. 通过 `scale_x/scale_y` 还原到原始帧坐标。

若没有候选点，会返回一个不可见 `TrackResult`：

```python
TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=max_heatmap_score)
```

## 6. BallTrackFilter 输入输出

`BallTrackFilter.update_candidates(...)` 输入：

- TrackNet 候选点列表。
- 当前帧尺寸。
- `court_prediction`。
- 当前稳定姿态的 `person_bboxes`。

输出：

```python
TrackResult(
    ball_xy=[x, y],
    visible=0 or 1,
    score=float,
    heatmap_shape=[h, w],
)
```

`visible=1` 表示当前滤波器认为该帧有可用球点。这个点可能来自真实候选点，也可能来自短时 coast 预测。

## 7. 候选点预过滤

`update_candidates(...)` 先做三件事：

1. **场地范围过滤**
   - 如果 `court_prediction` 有 `corners`，使用图像中的场地投影和向上空气区域过滤。
   - 如果没有 `corners` 但有 `image_to_court_h`，将图像点投影到场地平面后过滤。
   - 场地尺寸常量为宽 `610 cm`、长 `1340 cm`。

2. **静态热点过滤**
   - 连续多帧停在很小范围内的高分点会被记为 static hotspot。
   - 常用于抑制 INT8 量化后固定位置反复冒出的假点。
   - 相关日志字段：`static_filtered_count`、`static_hotspot_count`。

3. **候选排序**
   - 未锁定时直接选择最高分候选。
   - 已锁定时根据预测点、候选分数、运动门控、热力图排序、人体遮挡惩罚综合打分。
   - `candidates` 调试字段会记录每个候选的坐标、分数、到预测点距离、gate 结果、遮挡标记和 rank。

## 8. BallTrackFilter 状态机

滤波器内部维护：

- `_last_point`：上一帧确认点。
- `_velocity`：当前速度估计。
- `_missed_frames`：连续未接收真实点数量。
- `_coast_frames`：连续预测补点数量。
- `_locked`：是否锁定轨迹。
- `_candidate`：等待确认的新轨迹候选。
- `_history`：用于抛物线拟合的真实检测历史。
- `_real_detections_since_relock`：relock 后连续真实检测数。
- `_top_exit_frames_remaining`：顶部出画抑制倒计时。

单帧决策顺序：

1. 如果正在顶部出画抑制，直接输出不可见。
2. 将候选转换为 measurement。正式 measurement 要求：
   - `visible=1`
   - `score >= min_confidence = 0.35`
   - 点在画面内
   - 点在场地过滤范围内
3. 如果正式 measurement 不可用，且满足软阈值条件，尝试使用 soft measurement。
4. 如果没有可用 measurement，执行 reject，并在允许时 coast。
5. 如果未锁定，进入 bootstrap：
   - `score >= strong_relock_confidence = 0.72` 且不在顶部边缘区域时直接锁定。
   - 否则需要候选连续确认，且候选簇最高分不低于 `bootstrap_confirm_min_confidence = 0.55`。
   - 如果确认窗口内候选 `x` 几乎固定且 `y` 波动很小，会判为 `static_bootstrap_candidate`，继续等待而不锁定。
6. 如果预测点已经出画，reject/coast。
7. 如果检测到向上出界后的反向假点，进入 top-exit 抑制。
8. 如果当前 measurement 是顶部边缘幻觉点，reject/coast。
9. 如果人体遮挡活跃且候选落在人体 bbox 内，reject；低分候选可用 `person_occlusion_prediction` coast，高分候选会直接输出不可见。
10. 如果通过主运动门控，accept。
11. 如果通过 close gate，accept，并在 Y 方向冲突时废弃抛物线历史。
12. 否则更新 relock candidate：
    - `impact_direction_change`
    - `high_score_fast_relock`
    - `stable_new_candidate`
13. 如果 relock 不成立，reject，并视情况用抛物线或人体遮挡预测补点。

## 9. 主运动门控

`_passes_gate(...)` 依次检查：

1. 预测点是否出画。
2. 从上一点到候选点的速度是否超过 `max_speed_px_per_sec = 12000`。
3. 候选点到预测点距离是否在动态门限内。
4. 是否通过抛物线门控。
5. 如果候选分数位于 `inertia_relax_confidence = 0.65` 到 `inertia_relax_max_confidence = 0.75` 之间，且距离预测点不超过 `base_gate_px * inertia_relax_prediction_gate_ratio`，会跳过惯性门控，避免减速/转向时把贴近预测点的中高分真实候选误拒。
6. 其他情况继续检查惯性门控。

动态门限由以下因素组成：

- `base_gate_px = 80`
- 当前速度对应的像素步长
- `missed_frames * missed_gate_growth_px`
- 候选分数 bonus
- 最大不超过 `max_gate_px = 360`

惯性门控会检查：

- 加速度是否过大。
- 是否明显反向。
- 横向偏差是否过大。

## 10. 抛物线预测

抛物线预测用于短时丢球和 outlier 填补。

启用条件：

- `parabola_enabled = True`
- 至少 `parabola_min_points = 4` 个历史点。
- relock 后至少 `parabola_min_real_detections = 3` 个连续真实检测。
- 历史跨度足够，运动距离不低于 `parabola_min_motion_px = 42`。
- 拟合 RMSE 不超过 `parabola_max_fit_rmse_px = 48`。
- 预测 gap 不超过 `parabola_max_gap_frames = 8`。

预测策略：

- 如果抛物线可用，`_predict(...)` 优先返回抛物线预测点。
- 否则使用线性速度外推。
- `close_prediction_motion_break` 时，如果抛物线 Y 方向和真实检测 Y 方向冲突，会清空历史并改用当前实测 Y 速度。

这样可以减少 relock 后旧抛物线继续拖拽轨迹的问题。

## 11. Coast 预测

当当前帧没有可靠真实点时，滤波器可能输出预测点。

优先级：

1. 有可用抛物线预测时，用抛物线预测。
2. 否则用线性速度外推。

普通 coast 默认最多 `max_coast_frames = 10` 帧，用来覆盖常见的漏检窗口。之后仍有 `max_missed_frames = 12` 的丢失缓冲，避免刚结束 coast 就立即 drop lock。人体遮挡活跃时：

- `person_occlusion_coast_frames = 7`
- `person_occlusion_min_speed_px_per_sec = 250`
- 如果人体框内存在分数 `> person_occlusion_suppress_coast_candidate_score = 0.55` 的候选，说明球可能已经在击球瞬间改变方向，此帧不再沿旧速度外推，直接输出不可见。

预测点仍必须满足：

- 不出画。
- 不超出场地允许区域。

coast 点的分数会衰减，日志中常见：

- `coast/parabola_prediction`
- `coast/person_occlusion_prediction`

## 12. Relock 机制

当候选点无法通过当前轨迹门控时，滤波器不会立即切换轨迹，而是先积累 `_candidate`。

候选确认逻辑：

- 新候选离旧候选超过 relock 距离时，重置候选。
- 否则平滑更新候选位置和最高分。
- 记录候选出现次数、方向一致性和最近确认窗口内的 measurement。

bootstrap 确认额外过滤：

- `bootstrap_static_max_x_span_px = 3.0`
- `bootstrap_static_max_y_std_px = 15.0`
- 当候选在确认窗口内 `x` 几乎固定且 `y` 标准差很小，会认为是网柱、边线等静态热点，不触发 `candidate_confirmed` 锁定。

触发 relock 的路径：

- `stable_new_candidate`：候选连续达到 `relock_confirm_frames = 3` 且分数足够。
- `high_score_fast_relock`：高分候选 `score > 0.72` 且连续 2 次方向一致。
- `impact_direction_change`：短时漏检后出现方向突变，并满足角度、分数和预测误差要求。

relock 时会先 `_drop_lock()` 清空旧锁定状态，再 accept 新 measurement。relock 后抛物线必须重新积累足够真实检测才会启用。

## 13. 顶部出画和顶部假点

顶部出画逻辑用于处理球从画面上方飞出后，热力图在顶部边缘产生假点的问题。

主要规则：

- 如果历史运动显示球高速向上并接近顶部，会进入 `top_exit_enter/likely_top_exit`。
- 进入顶部出画后，会清空锁定并抑制若干帧。
- 如果当前候选在顶部边缘，但与预测/上一点存在大距离落差，会判为 `top_edge_hallucination`。

相关字段：

- `top_exit_remaining`
- `reason=likely_top_exit`
- `reason=top_edge_hallucination`
- `reason=measurement_reverses_after_top_exit`

## 14. 人体遮挡处理

人体遮挡来自姿态跟踪后的 `person_bboxes`。

滤波器使用人体 bbox 做两类判断：

1. 轨迹线段是否穿过人体 bbox，用于判断遮挡是否可能发生。
2. 当前候选是否落在人体 bbox 内，用于拒绝人体区域内的假球点。

人体遮挡活跃时，会倾向于：

- 给落在人体 bbox 内的候选增加惩罚。
- 拒绝 `person_occlusion_candidate`。
- 延长 coast 帧数，用 `person_occlusion_prediction` 保持轨迹连续。
- 但如果人体 bbox 内被拒绝的候选分数 `> 0.55`，会拒绝旧速度 coast，避免击球瞬间沿击球前方向漂移出假轨迹。

## 15. 轨迹渲染

`TrackTrailRenderer` 不参与 BallTrackFilter 的数据滤波，但参与两件事：

- 绘制球点和尾迹。
- 根据滤波后的轨迹和姿态生成 `hit_event`。

尾迹规则：

- 只绘制最近 `history_seconds = 0.5` 秒。
- 不同 segment 之间不连线。
- 相邻点距离超过 `trail_break_threshold_px = 80` 时不连线。

因此 relock 时即使数据存在跳变，视觉尾迹也会自然断开，避免画出穿越半个画面的长线。

## 16. 击球点检测

`TrackTrailRenderer.update_hit_detection(...)` 会为每个可见球点保存：

```text
(timestamp_s, frame_id, x, y, score, segment_id, occluded, pose_score)
```

segment 会在以下情况增加：

- 当前帧不可见。
- 两个可见点时间间隔超过 `hit_max_gap_seconds = 0.16`。

击球检测基于最近 3 个可见点：

```text
prev -> mid -> current
```

基础过滤：

- 至少 3 点。
- 前后时间差都大于 0。
- 前后位移都不小于 3 px。
- 最大速度达到阈值：
  - 普通：`hit_min_speed_px_per_sec = 500`
  - 有姿态辅助：`hit_pose_assist_relaxed_min_speed_px_per_sec = 360`
- 排除顶部出画反转。
- 排除地板弹跳。
- 排除人体遮挡造成的轨迹突变，除非姿态 override 足够强。

可靠性规则：

- 同段普通击球要求 `prev/mid/current` 最低分数不小于 `hit_min_track_score = 0.25`。
- 同段突变命中要求：
  - `mid.score >= hit_abrupt_min_score = 0.50`
  - `current.score >= hit_abrupt_min_score = 0.50`
  - `dist_before >= hit_abrupt_min_jump_px = 120`
  - `dist_before >= hit_abrupt_large_jump_px = 270`，或 `dist_before >= dist_after * hit_abrupt_min_jump_ratio = 2.5`
- 跨段突变命中会寻找旧 segment 中的 anchor：
  - anchor 距离 `mid` 不超过 `hit_cross_segment_anchor_max_gap_seconds = 0.22`
  - anchor 分数不低于 `hit_min_track_score = 0.25`
  - anchor 到 `mid` 的跳变满足突变距离，且满足大跳变阈值或比例要求

形状规则：

- 方向突变：`turn_deg >= hit_min_turn_deg = 85`
- 速度突变：`turn_deg >= 45` 且 `speed_change >= 1.7`
- 姿态辅助：`pose_score >= 0.60` 且满足较宽松转角（`>= 55`），或速度变化（`>= 1.25`）加最低转角（`>= 20`）

提交规则：

- 命中点会记录为真实触发点的 `frame_id/timestamp_ms/ball_xy`，不是确认帧。
- 两次击球点之间有 `hit_cooldown_seconds = 0.18` 的冷却时间。
- 红色 marker 显示 `hit_marker_seconds = 2.0` 秒。

当前需要注意：姿态辅助路径仍可能在真实击球后一段距离触发弱速度变化，因此调试击球点误检时应重点看 `pose_score`、`turn_deg` 和 `speed_change`。

## 17. 姿态辅助击球评分

姿态辅助在 `TrackTrailRenderer` 内完成，使用左右手腕、手肘、肩膀关键点。

单个手臂评分由三部分组成：

- 手腕到球的距离：距离越近越高，最大距离 `130 px`。
- 手腕速度：阈值参考 `hit_pose_assist_min_wrist_speed_px_per_sec = 220`。
- 手臂伸展程度：肩膀到手腕距离 / 上臂加前臂长度。

综合权重：

```text
pose_score = 0.40 * proximity + 0.45 * wrist_speed + 0.15 * extension
```

这个分数只用于降低击球点检测门槛或覆盖遮挡/弹跳过滤，不会反向修改 BallTrackFilter 输出的球轨迹。

## 18. FrameResult

每一帧最终形成统一结构：

```python
FrameResult(
    frame_id=frame_id,
    pose=last_pose,
    track=track,
)
```

其中：

- `pose` 是经过 `CourtPoseTargetTracker` 稳定后的姿态列表。
- `track` 是 `BallTrackFilter` 输出的最终球轨迹点。

后续 UI、日志、击球点检测、BST 动作识别都基于 `FrameResult`。

## 19. 球员位移统计

球员位移统计不影响球轨迹，只用于 UI 指标。

流程：

1. 从稳定后的姿态中取球员 bbox / 关键点。
2. 使用 `court_prediction.image_to_court_h` 投影到真实场地平面。
3. `PlayerDistanceAccumulator` 按球员 id 或稳定索引累计场地平面位移。
4. UI 显示上方球员、下方球员的累计米数。

如果当前没有可用场地投影，会保持已有累计值，但重置当前跟踪点，避免下一次重新检测时跨大距离累加。

## 20. 日志输出

在 UI 中打开 `Debug Logs / Write frame analysis logs` 后，会在以下目录输出：

```text
outputs/pyqt_debug/
```

一组日志包含：

- `*_track_debug.csv`
- `*_frame_log.jsonl`

### track_debug.csv

CSV 来自 `BallTrackFilter.last_debug_record()`。

关键字段：

- `frame_index`：滤波器处理帧序号。
- `action`：当前帧滤波动作。
- `reason`：动作原因。
- `raw_candidate_count`：TrackNet 原始候选数量。
- `candidate_count`：预过滤后的候选数量。
- `selected_candidate_index`：最终选择候选索引。
- `selected_candidate_rank`：候选排序分数或 primary。
- `input_x/input_y/input_score`：送入状态机的候选点。
- `output_x/output_y/output_score`：滤波器输出点。
- `locked_before/locked_after`：处理前后是否锁定。
- `missed_before/missed_after`：处理前后连续丢失帧数。
- `coast_before/coast_after`：处理前后连续预测帧数。
- `pred_x/pred_y`：处理前预测位置。
- `velocity_x_before/velocity_y_before`：处理前速度。
- `velocity_x_after/velocity_y_after`：处理后速度。
- `top_exit_remaining`：顶部出画抑制剩余帧数。
- `candidates`：每个候选的坐标、分数、距离、gate、遮挡和 rank。

常见 `action/reason`：

- `bootstrap_wait/waiting_for_candidate_confirmation`
- `bootstrap_wait/static_bootstrap_candidate`
- `bootstrap_accept/candidate_confirmed`
- `accept/passes_motion_gate`
- `accept/soft_confidence_motion_gate`
- `accept/close_prediction_motion_break`
- `relock_accept/stable_new_candidate`
- `relock_accept/high_score_fast_relock`
- `relock_accept/impact_direction_change`
- `coast/parabola_prediction`
- `coast/person_occlusion_prediction`
- `reject/candidate_failed_motion_gate`
- `reject/person_occlusion_candidate`
- `reject/person_occlusion_candidate_high_score`
- `reject/top_edge_hallucination`
- `top_exit_enter/likely_top_exit`
- `drop_lock/max_missed_frames`

### frame_log.jsonl

JSONL 来自 `frame_result_log_record(...)`，每一行是一帧：

```json
{
  "frame_id": 123,
  "timestamp_ms": 2050,
  "ball": {
    "xy": [100.0, 200.0],
    "visible": 1,
    "score": 0.72
  },
  "pose": [
    {
      "person_id": 0,
      "bbox": [10.0, 20.0, 100.0, 220.0],
      "person_score": 0.91,
      "keypoints": [[50.0, 80.0]],
      "keypoint_scores": [0.9]
    }
  ],
  "hit_event": {
    "frame_id": 123,
    "timestamp_ms": 2050,
    "ball_xy": [100.0, 200.0]
  }
}
```

如果该帧没有击球事件，`hit_event` 为 `null`。

排查轨迹误检时优先看 `track_debug.csv`；排查击球点漏检或误检时优先看 `frame_log.jsonl`，并结合 `track_debug.csv` 中相同时间附近的 `action/reason`。

## 21. 调试建议

### 轨迹乱飘

优先检查：

1. `track_debug.csv` 中是否出现 `relock_accept`。
2. relock 前后 `pred_x/pred_y` 与候选点距离。
3. `candidates` 中是否有多个高分候选。
4. 是否频繁出现 `coast/parabola_prediction`。
5. 是否存在 `top_edge_hallucination` 或 `person_occlusion_candidate`。
6. 人体遮挡帧内是否出现 `reject/person_occlusion_candidate_high_score`，这表示高分候选被人体框压制且没有继续 coast。

### 击球点漏检

优先检查：

1. `frame_log.jsonl` 中真实拐点附近的三点轨迹。
2. 对应 `track_debug.csv` 是否处于 `coast`、`relock_accept` 或跨段状态。
3. 拐点是否被 `score`、`segment`、`cross_no_anchor`、`top_exit`、`floor_bounce`、`person_occlusion` 或 `cooldown` 过滤。
4. `pose_score` 是否足够，但转角和速度变化是否太弱。

### 击球后出现第二个假红点

优先检查：

1. 两个 `hit_event` 的时间间隔是否刚好超过 `hit_cooldown_seconds = 0.18`。
2. 第二个红点是否来自姿态辅助路径。
3. 第二个红点处 `turn_deg` 是否很小但 `speed_change` 触发。
4. 第二个红点附近是否是 relock 后连续稳定飞行段，而不是真实击球瞬间。

## 22. CLI Runner

除 PyQt6 外，项目中还有多个 CLI runner：

- `src/runners/track_video_runner.py`
- `src/runners/pose_video_runner.py`
- `src/runners/unified_runner.py`
- `src/runners/tracknet_realtime_runner.py`

这些 runner 复用 `TrackBranch`、`PoseBranch`、`BallTrackFilter`、`TrackTrailRenderer` 等核心模块，适合离线批处理或单模块验证。PyQt6 的实时体验逻辑额外包含 UI 刷新节流、调试 JSONL、BST 击球识别和球员位移统计。
