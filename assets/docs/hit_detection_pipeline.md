# 击球点判定流程

更新时间：2026-05-03

本文档只描述当前击球点 `hit_event` 的判定流程。轨迹模型、轨迹滤波、relock、coast 等上游流程见 `assets/docs/current_prediction_pipeline.md`。

主要实现文件：

- `src/utils/visualize.py`
- `apps/pyqt6/controllers/analysis_controller_runtime.py`
- `src/utils/exporters.py`

## 1. 输入和输出

击球点判定发生在 `TrackTrailRenderer` 内。输入是每帧的 `FrameResult`：

```python
FrameResult(
    frame_id=frame_id,
    pose=last_pose,
    track=track,
)
```

其中：

- `track` 是 `BallTrackFilter` 输出后的最终球点，不是 TrackNet 原始候选点。
- `pose` 是 `CourtPoseTargetTracker` 稳定后的球员姿态。

输出是 `hit_event`：

```json
{
  "frame_id": 123,
  "timestamp_ms": 2050,
  "ball_xy": [100.0, 200.0]
}
```

如果当前帧没有新击球点，`last_hit_event()` 返回 `None`。

## 2. 调用位置

PyQt6 视频和摄像头流程都会调用：

```python
trail_renderer.draw_on(...)
```

或在当前帧不需要 UI 渲染时调用：

```python
trail_renderer.update_hit_detection(...)
```

因此击球点检测不依赖 UI 是否刷新。即使显示帧率被限制，`hit_event` 仍会按推理帧持续更新。

## 3. 可见球点缓存

每当 `result.track.visible` 为真时，`TrackTrailRenderer` 会保存一个内部点：

```text
(timestamp_s, frame_id, x, y, score, segment_id, occluded, pose_score)
```

字段含义：

- `timestamp_s`：当前帧时间，优先来自 `timestamp_ms`。
- `frame_id`：视频帧编号或摄像头处理帧编号。
- `x/y`：滤波后球点坐标。
- `score`：滤波后球点分数。
- `segment_id`：轨迹片段编号。
- `occluded`：球点是否落在人体 bbox 内。
- `pose_score`：姿态辅助击球分数。

只保留最近 `history_seconds = 0.5` 秒的点。

## 4. Segment 切分

击球检测会区分同一段轨迹和跨段轨迹。

`segment_id` 在两种情况下增加：

- 当前帧球不可见。
- 两个可见球点之间的时间间隔超过 `hit_max_gap_seconds = 0.16` 秒。

这意味着 relock、长时间漏检、球出画后再出现，通常会进入新 segment。

## 5. 姿态辅助分数

姿态辅助来自左右手臂关键点：

```text
左臂: shoulder=5, elbow=7, wrist=9
右臂: shoulder=6, elbow=8, wrist=10
```

关键点最低分数：

- wrist：`0.20`
- elbow：`0.15`
- shoulder：`0.15`

每个手腕会计算：

- 手腕到球的距离。
- 手腕相对上一帧的速度。
- 手臂伸展程度。

单臂得分：

```text
pose_score = 0.40 * proximity_score
           + 0.45 * speed_score
           + 0.15 * extension_score
```

参数：

- `hit_pose_assist_max_ball_wrist_px = 130`
- `hit_pose_assist_min_wrist_speed_px_per_sec = 220`

最终 `pose_score` 取所有手臂中的最大值。

## 6. 三点判定窗口

击球点检测只看最近 3 个可见球点：

```text
prev -> mid -> current
```

当前实现的核心假设是：击球点通常发生在 `mid`，因为只有看到 `current` 后，才能确认 `prev -> mid -> current` 的方向或速度变化。

跨段突变时，命中点有可能不是 `mid`，而是旧 segment 中的 anchor 点。详见“跨段突变判定”。

## 7. 基础运动量计算

对三点计算：

```text
dt_before = mid.time - prev.time
dt_after  = current.time - mid.time

v_before = mid.xy - prev.xy
v_after  = current.xy - mid.xy

dist_before = length(v_before)
dist_after  = length(v_after)

speed_before = dist_before / dt_before
speed_after  = dist_after / dt_after
```

如果不满足以下条件，直接不判定击球：

- 点数少于 3。
- `dt_before <= 0` 或 `dt_after <= 0`。
- `min(dist_before, dist_after) < 3 px`。

## 8. 速度门槛

速度门槛取决于姿态辅助：

- 普通情况：`hit_min_speed_px_per_sec = 500`
- 有姿态辅助：`hit_pose_assist_relaxed_min_speed_px_per_sec = 360`

如果：

```text
max(speed_before, speed_after) < min_speed
```

则不判定击球。

姿态辅助成立条件：

```text
pose_score >= hit_pose_assist_score = 0.60
```

姿态 override 条件：

```text
pose_score >= hit_pose_assist_override_score = 0.50
```

override 用于允许真实挥拍动作覆盖部分弹跳/遮挡过滤。

## 9. 假点过滤

进入形状判定前，会先排除几类常见假阳性。

### 顶部出画反转

`_looks_like_top_exit(...)` 用于抑制球从画面上方飞出后，顶部附近热力图假点造成的红点。

顶部区域：

```text
top_band = max(hit_top_exit_band_px, frame_height * hit_top_exit_band_ratio)
         = max(36 px, frame_height * 0.08)
```

如果三点中有点接近顶部，且 `mid` 位于顶部区域，并且 `prev -> mid` 是明显向上运动，则过滤。

### 地板弹跳

`_looks_like_floor_bounce(...)` 用于抑制球落地反弹被误判为击球。

主要条件：

- `prev -> mid` 明显向下。
- `mid -> current` 明显向上。
- `mid` 是三点中最低点。
- 前后运动都有足够大的垂直分量。
- 反弹速度没有明显超过下落速度。

相关参数：

- `hit_floor_bounce_min_vertical_px = 10`
- `hit_floor_bounce_min_vertical_ratio = 0.45`
- `hit_floor_bounce_max_rebound_speed_ratio = 1.35`

如果姿态 override 不成立，地板弹跳会被过滤。

### 人体遮挡

如果 `mid` 落在人体 bbox 内，并且 `prev` 或 `current` 也处于人体遮挡状态，则认为该三点形状可能来自人体遮挡假点。

如果姿态 override 不成立，人体遮挡会被过滤。

## 10. 轨迹可靠性

基础可靠性要求：

```text
min(prev.score, mid.score, current.score) >= hit_min_track_score = 0.25
```

如果三点同 segment，但不满足可靠性要求，也不是同段突变命中，则不判定击球。

这个规则主要用于过滤 coast 低分点、弱预测点、遮挡预测点造成的假红点。

## 11. 同段突变判定

同段突变用于识别 relock 或轨迹跳变附近的明确拐点。

条件：

```text
same_segment = prev.segment == mid.segment == current.segment
mid.score >= hit_abrupt_min_score = 0.50
current.score >= hit_abrupt_min_score = 0.50
dist_before >= hit_abrupt_min_jump_px = 120
dist_before >= hit_abrupt_large_jump_px = 270
    OR dist_before >= dist_after * hit_abrupt_min_jump_ratio = 2.5
```

满足后直接把 `mid` 作为击球点提交。

## 12. 跨段突变判定

跨段突变用于处理漏检或 relock 后，`prev` 在旧 segment，`mid/current` 在新 segment 的情况：

```text
cross_segment_abrupt = prev.segment != mid.segment
                    and mid.segment == current.segment
```

跨段时不会直接把 `mid` 判为击球点，而是尝试在旧 segment 中寻找 anchor。

anchor 条件：

- anchor 不属于 `mid/current` 的 segment。
- `mid.time - anchor.time <= hit_cross_segment_anchor_max_gap_seconds = 0.22`
- `anchor.score >= hit_min_track_score = 0.25`
- `mid.score >= hit_abrupt_min_score = 0.50`
- `current.score >= hit_abrupt_min_score = 0.50`
- `distance(anchor, mid) >= hit_abrupt_min_jump_px = 120`
- `distance(anchor, mid) >= hit_abrupt_large_jump_px = 270`，或 `distance(anchor, mid) >= dist_after * hit_abrupt_min_jump_ratio = 2.5`

如果找到 anchor，提交 anchor 作为击球点。这样红点更接近真实拐点，而不是 relock 后的确认帧。

如果不是同 segment，且找不到跨段 anchor，则不判定击球。

## 13. 方向和速度形状判定

若没有被同段/跨段突变提前命中，会继续计算转角和速度变化：

```text
turn_deg = angle(v_before, v_after)
speed_change = max(speed_before, speed_after) / min(speed_before, speed_after)
```

普通方向突变：

```text
turn_deg >= hit_min_turn_deg = 85
```

速度突变：

```text
turn_deg >= hit_speed_change_min_turn_deg = 45
and speed_change >= hit_min_speed_change_ratio = 1.7
```

姿态辅助突变：

```text
pose_score >= hit_pose_assist_score = 0.60
and (
    turn_deg >= hit_pose_assist_relaxed_turn_deg = 55
    or (
        speed_change >= hit_pose_assist_relaxed_speed_change_ratio = 1.25
        and turn_deg >= hit_pose_assist_speed_change_min_turn_deg = 20
    )
)
```

三种形状任意一种成立，就把 `mid` 作为击球点提交。

## 14. 提交和冷却

提交击球点时调用 `_commit_hit_detection(...)`。

冷却时间：

```text
hit_cooldown_seconds = 0.18
```

如果当前候选击球点距离上一次击球点不足 0.18 秒，则丢弃。

提交成功后：

1. 更新 `_last_hit_time_s`。
2. 返回 `_HitDetection(timestamp_s, frame_id, x, y)`。
3. `update_hit_detection(...)` 写入 `_last_hit_event`。
4. 红色 marker 加入 `_hit_markers`。

注意：`hit_event` 使用的是被提交点本身的时间和坐标，不是当前确认帧的时间和坐标。

## 15. 红点显示

红色击球点 marker 参数：

- `hit_marker_radius = 7`
- `hit_marker_seconds = 2.0`

marker 会显示 2 秒，然后被 `_prune(...)` 移除。

轨迹尾迹和击球点 marker 是两个概念：

- 尾迹用于显示最近轨迹。
- marker 用于显示已确认击球点。

尾迹断线规则不会改变 `hit_event` 数据。

## 16. 与轨迹尾迹的关系

轨迹尾迹绘制也在 `TrackTrailRenderer` 中，但它只影响视觉效果。

尾迹不连线的情况：

- 相邻点属于不同 segment。
- 相邻点距离超过 `trail_break_threshold_px = 80`。

这可以避免 relock 时画出穿越画面的长线，但不会改变击球点判定本身。

## 17. 日志中的表现

`frame_log.jsonl` 会记录 `hit_event`：

```json
"hit_event": {
  "frame_id": 123,
  "timestamp_ms": 2050,
  "ball_xy": [100.0, 200.0]
}
```

如果没有新击球点：

```json
"hit_event": null
```

当前 JSONL 不直接记录击球点被哪条规则过滤。排查时需要结合：

- `frame_log.jsonl` 中的球点、姿态、`hit_event`。
- `track_debug.csv` 中同时间附近的 `action/reason`。
- 必要时离线复算 `prev/mid/current` 的 `turn_deg`、`speed_change`、`pose_score`、segment。

## 18. 常见漏检原因

### 轨迹断段但找不到 anchor

表现：

- 真实拐点发生在旧 segment 末尾。
- relock 后 `mid/current` 高分稳定。
- 但旧 anchor 太远、分数太低，或跳变比例不足。

结果：`cross_segment_anchor` 为空，跨段击球点被过滤。

### 低分 coast 点参与三点窗口

表现：

- `prev` 或 `mid` 是低分预测点。
- 三点最低分低于 `hit_min_track_score = 0.25`。

结果：同段普通击球被可靠性规则过滤。

### 冷却时间过滤

表现：

- 一个真实击球点后 0.18 秒内又出现突变。

结果：第二个候选被 `_commit_hit_detection(...)` 丢弃。

这通常用于抑制同一次击球后的重复红点，但也可能过滤极少数非常接近的真实动作。

## 19. 常见误检原因

### 击球后一段距离出现第二个红点

可能原因：

- 第一个真实击球点后，轨迹仍有速度突变。
- 0.18 秒冷却结束后，姿态辅助路径再次满足弱速度变化。
- `pose_score >= 0.60` 且 `speed_change >= 1.25`。

排查重点：

- 第二个红点处的 `turn_deg` 是否很小。
- `speed_change` 是否刚超过 1.25。
- `pose_score` 是否来自球经过人体附近，而不是真实挥拍。

### 球落地弹跳被判击球

当前已有地板弹跳过滤，但如果姿态 override 同时成立，弹跳过滤会被放开。

排查重点：

- `mid` 是否为三点最低点。
- `vy_before` 是否向下，`vy_after` 是否向上。
- 附近是否有手腕关键点高速经过球点。

### 人体遮挡处产生红点

当前人体遮挡过滤只看球点是否落在人体 bbox 内，不能理解真实的前后遮挡关系。

排查重点：

- `mid.occluded` 是否为真。
- `prev/current` 是否也处于人体 bbox 内。
- 姿态 override 是否把遮挡过滤放开。

## 20. 参数速查

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `history_seconds` | `0.5` | 击球检测和尾迹缓存时长 |
| `hit_min_speed_px_per_sec` | `500` | 普通速度门槛 |
| `hit_pose_assist_relaxed_min_speed_px_per_sec` | `360` | 姿态辅助速度门槛 |
| `hit_min_turn_deg` | `85` | 普通方向突变角度 |
| `hit_speed_change_min_turn_deg` | `45` | 速度突变最低角度 |
| `hit_min_speed_change_ratio` | `1.7` | 普通速度突变比例 |
| `hit_cooldown_seconds` | `0.18` | 两次击球点最小间隔 |
| `hit_max_gap_seconds` | `0.16` | 超过该间隔切新 segment |
| `hit_top_exit_band_px` | `36` | 顶部出画过滤像素带 |
| `hit_top_exit_band_ratio` | `0.08` | 顶部出画过滤高度比例 |
| `hit_pose_assist_score` | `0.60` | 姿态辅助阈值 |
| `hit_pose_assist_override_score` | `0.50` | 姿态 override 阈值 |
| `hit_pose_assist_max_ball_wrist_px` | `130` | 球到手腕最大辅助距离 |
| `hit_pose_assist_min_wrist_speed_px_per_sec` | `220` | 手腕速度参考阈值 |
| `hit_pose_assist_relaxed_turn_deg` | `55` | 姿态辅助转角阈值 |
| `hit_pose_assist_relaxed_speed_change_ratio` | `1.25` | 姿态辅助速度变化阈值 |
| `hit_pose_assist_speed_change_min_turn_deg` | `20` | 姿态辅助速度变化分支的最低转角，防止直线飞行中速度波动触发假阳性 |
| `hit_floor_bounce_min_vertical_px` | `10` | 弹跳过滤最小垂直位移 |
| `hit_floor_bounce_min_vertical_ratio` | `0.45` | 弹跳过滤垂直分量比例 |
| `hit_floor_bounce_max_rebound_speed_ratio` | `1.35` | 弹跳过滤反弹速度上限 |
| `hit_min_track_score` | `0.25` | 三点可靠性最低分 |
| `hit_abrupt_min_score` | `0.50` | 突变击球点最低分 |
| `hit_abrupt_min_jump_px` | `120` | 突变跳变距离 |
| `hit_abrupt_min_jump_ratio` | `2.5` | 突变前后距离比例 |
| `hit_abrupt_large_jump_px` | `270` | 大跳变直接判定阈值，超过后不再要求比例 |
| `hit_cross_segment_anchor_max_gap_seconds` | `0.22` | 跨段 anchor 最大时间间隔 |
| `trail_break_threshold_px` | `80` | 视觉尾迹断线阈值 |

## 21. 调试建议

如果红点晚于真实拐点：

1. 查看真实拐点是否发生在 segment 边界。
2. 检查旧 segment 是否存在可用 anchor。
3. 检查拐点附近是否有低分 coast 点导致可靠性过滤。
4. 对比 `hit_event.frame_id` 和当前确认帧，确认是否只是确认延迟。

如果击球点漏检：

1. 从 `frame_log.jsonl` 取真实拐点前后 3 到 5 个可见球点。
2. 计算三点 `turn_deg`、`speed_change`、`pose_score`。
3. 检查是否被顶部出画、地板弹跳、人体遮挡过滤。
4. 检查是否跨段且 `cross_segment_anchor` 找不到。
5. 检查是否在上一次 `hit_event` 后 0.18 秒内。

如果击球后出现第二个假红点：

1. 看第二个红点和第一个红点间隔是否刚超过 0.18 秒。
2. 看第二个红点是否主要由姿态辅助触发。
3. 如果 `turn_deg` 很小但 `speed_change` 刚过 1.25，应重点怀疑姿态辅助过宽。
4. 检查该段轨迹是否经过人体 bbox 或发生 relock。
