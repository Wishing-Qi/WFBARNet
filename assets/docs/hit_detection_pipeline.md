# 击球与轨迹事件检测流程

当前主击球事件由 `src/postprocess/trajectory_events.py` 中的 `RealtimeTrajectoryEventDetector` 生成。旧的 `TrackTrailRenderer` 内部击球检测入口已经清理；可视化层只负责轨迹尾迹和事件 marker，不再生成 `hit_event`。

## 数据流

```text
TrackNet 原始候选
  -> TrackNetV3TrajectoryFilter
  -> FrameResult(track=filtered_track, pose=stable_pose)
  -> RealtimeTrajectoryEventDetector.update(...)
  -> trajectory_event
  -> hit_event / landing_event
  -> JSONL 日志、UI marker、BSTStrokeRecognizer
```

`trajectory_event` 会记录任意轨迹事件；当 `event_type == "hit"` 时复制为主 `hit_event`，当 `event_type == "landing"` 时复制为 `landing_event`。

## 击球规则

主击球只接受反转类规则：

- `vy_reversal`
- `vx_reversal`

以下规则只作为辅助证据或其它事件来源，不会单独生成主击球点：

- `acceleration_peak`
- `speed_local_max`
- `y_local_max`

当前主击球还会做有效性过滤：

- 当前候选点 `score >= 0.48`
- 相邻可见点 `score >= 0.35`
- 反转后速度在 `8.0` 到 `220.0` 像素/帧之间
- 反转幅度 `>= 8.0`
- 顶部忽略区为 `max(36 px, frame_height * 0.08)`
- 候选确认延迟不能超过 `12` 帧
- 同类事件冷却时间为 `0.18` 秒

`acceleration_peak` 曾经造成大量弱峰值误检，所以现在只会提升已有反转击球的置信度，不再独立触发红色击球点。

## 落地与出画

落地点事件来自：

- `speed_step`
- `low_speed_start`
- `speed_drop`
- `trajectory_end`
- `tracking_lost_rally_end`

速度类落地规则只在连续可见窗口内生效，避免短暂丢球后的 `[-1,-1]` 缺失点污染速度并制造假减速。低速类落地规则会忽略画面顶部高空区；如果球还在画面上半部且最近一段窗口仍呈向上运动，也不会把低速点当作落地，避免把高远球顶点误判为死球。`trajectory_end` 只在轨迹尾部已经进入低速段时触发，不再把高速丢球尾点兜底为落地。

`tracking_lost_rally_end` 是延迟确认规则：非顶部/非边缘 tracking lost 持续达到阈值，且丢失前速度不像离群跳点时，才把该丢失点确认为回合结束。

出画事件来自：

- `visibility_drop_edge`
- `visibility_drop_upward`
- `visibility_drop_high_altitude`
- `visibility_drop_tracking_lost`

普通 `visibility_drop` 不再生成落地事件，而是标记为 `visibility_drop_tracking_lost`。出画事件用于标记跟踪丢失或画面边缘离开，不单独判定回合结束；回合状态目前只由 `landing` 事件结束。

## 可视化

`TrackTrailRenderer` 绘制最近 `history_seconds = 0.5` 秒的轨迹。不同 segment 或相邻点距离超过 `trail_break_threshold_px = 80` 时不连线。

事件 marker 颜色：

- `hit`：红色圆点
- `landing`：绿色菱形
- `out_of_frame`：紫色叉号

marker 显示时长由 `event_marker_seconds = 2.0` 控制。

## 日志排查

`outputs/pyqt_debug/*_frame_log.jsonl` 中相关字段：

- `hit_event`：当前帧新确认的主击球事件，没有则为 `null`
- `trajectory_event`：当前帧新确认的任意轨迹事件
- `landing_event`：当前帧新确认的落地事件

常看字段：

- `rule`
- `confidence`
- `all_rules`
- `auxiliary_rules`
- `features`
- `ball.score`

如果误检增多，优先检查是否出现大量非反转规则或低分反转；如果漏检增多，优先检查 `score`、顶部忽略区、邻居点分数和 `max_event_lag_frames`。
