# 回合判断与统计算法说明

本文档说明当前项目是否有回合判断算法、`D:\Github\monotrack` 的相关做法，以及 WFBARNet 当前实际使用的回合状态与统计口径。

## 结论

当前 WFBARNet 有回合级统计与状态判断模块，但没有完整的多回合自动切分算法。

- 已有模块：`src/postprocess/rally_stats.py` 中的 `RallyStatsAccumulator`。
- 当前定位：把当前加载的视频或分析片段当作一个回合/片段来累计统计。
- 回合开始：优先取第一段稳定可见且有位移的球飞行时间；如果没有稳定飞行段但先收到 `hit` 事件，则用 `hit` 时间初始化。
- 回合结束：收到可信 `landing` 轨迹事件后，`rally_state` 变为 `回合结束`。
- `out_of_frame` 只记录跟踪丢失/出画次数，不单独结束回合；日志显示高远球从画面顶部短暂消失后经常会重新出现。
- 未实现能力：不根据发球、比分牌、裁判信号、镜头切换或长时间静止来自动切出多个回合；回合结束后也不会自动重置并开启下一回合。

也就是说，当前算法能回答“这个分析片段内的回合状态和统计是多少”，还不能回答“完整比赛视频里每个回合从哪一帧开始、哪一帧结束”。

## MonoTrack 做法

查看 `D:\Github\monotrack` 后，相关结论如下。

### 输入假设

MonoTrack 的主处理流程默认输入已经是单个回合视频：

- `setup/setup.py` 遍历 `{data_dir}/{match}/rally_video/*.mp4`，对每个 rally 视频分别做球场、姿态和球轨迹处理。
- `python/ai-badminton/src/ai_badminton/pipeline_clean.py` 中的 `run_hit_detection(...)` 和 `run_3d_trajectory_reconstruction(...)` 也都遍历 `match_path / "rally_video"`。

因此 MonoTrack 主流程没有在这里从完整比赛视频实时判定多个回合边界。

### 切分来源

仓库里有一个人工切分脚本：

- `deprecated/scripts/dataset/split_videos.py`

它读取 VIA CSV 中的片段起止时间，然后用 ffmpeg 切出 `rally_{idx}.mp4`。这说明 MonoTrack 数据准备阶段主要依赖人工标注好的回合起止时间。

### 实验性回合切分 notebook

仓库里还有：

- `notebooks/rally-segmenter.ipynb`

该 notebook 使用 OpenPose 提取单人关键点，并计算相邻帧人体中心点的运动距离曲线：

```text
centroid(keypoints) -> frame-to-frame centroid distance -> plot
```

它更像探索“人体运动强弱是否能辅助切分回合”的实验笔记，没有形成可直接复用的生产级回合边界判定器。

### 击球与轨迹重建

MonoTrack 真正成熟的时序逻辑集中在击球和轨迹重建：

- `python/ai-badminton/src/ai_badminton/hit_detector.py`
  - `AdhocHitDetector` 用球轨迹 X/Y 的局部极值和斜率阈值找击球候选，再用姿态可达性判断击球球员。
  - `MLHitDetector` 使用 HitNet 模型输出三分类概率，并通过后处理或动态规划约束击球间隔与双方交替。
- `python/ai-badminton/src/ai_badminton/rally_reconstructor.py`
  - 读取 hit 序列后，把相邻两次击球之间作为一段飞行轨迹做 3D 重建。

这些逻辑处理的是已切好的 rally 内部事件，不负责从完整比赛视频切分 rally。

## WFBARNet 当前做法

### 数据流

当前 PyQt 实时播放、批量分析和离线处理路径都会创建 `RallyStatsAccumulator`：

- `apps/pyqt6/controllers/analysis_controller_runtime.py`

核心数据流为：

```text
TrackNet 候选
  -> TrackNetV3TrajectoryFilter
  -> RealtimeTrajectoryEventDetector
  -> hit / landing / out_of_frame
  -> RallyStatsAccumulator.add_trajectory_event(...)

Pose + court_prediction
  -> project_player_points_to_court(...)
  -> RallyStatsAccumulator.update_frame(...)

BSTStrokeRecognizer
  -> add_bst_prediction(...)
  -> 合并击球类型、球员、区域
```

### 回合开始

`RallyStatsAccumulator.update_frame(...)` 每帧更新基础统计：

- `_start_timestamp_ms`：第一帧时间。
- `_last_timestamp_ms`：最近处理帧时间。
- `_rally_start_timestamp_ms`：第一段稳定球飞行或第一条 `hit` 事件的时间。

稳定球飞行的默认口径：

- 连续可见球点达到 `5` 帧。
- 这段球点相对起点的最大位移达到 `30 px`。
- 这段球点平均置信度不低于 `0.40`。

这样可以避免 output 日志里第一帧偶发球点把回合误判为从 `0 ms` 开始。如果没有稳定飞行段但先收到 `hit` 事件，`add_trajectory_event(...)` 会用该 `hit` 时间初始化 `_rally_start_timestamp_ms`。`out_of_frame` 不会启动回合。

### 回合结束

`add_trajectory_event(...)` 只接受三类事件：

- `hit`
- `landing`
- `out_of_frame`

其中：

- `hit` 会创建或更新击球记录，不会结束回合。
- `landing` 会增加 `landing_count`，并把 `_rally_end_timestamp_ms` 设为事件时间。
- `out_of_frame` 会增加 `out_of_frame_count`，但不会写入 `_rally_end_timestamp_ms`。

`summary()` 中的状态规则很直接：

```text
if _rally_start_timestamp_ms is None:
    rally_state = "未开始"
elif _rally_end_timestamp_ms is not None:
    rally_state = "回合结束"
else:
    rally_state = "回合中"
```

所以当前“判回合结束”完全依赖轨迹事件检测器是否产生可信的 `landing`。`out_of_frame` 可以辅助排查 TrackNet 丢失、顶部出画或高远球离开画面的情况，但不再作为死球条件。

为降低 output 日志里出现的提前结束问题，轨迹事件检测器当前还做了两类保护：

- `trajectory_end` 需要连续缺失达到阈值，并且最后可见尾部已经进入低速段；不会再把高速尾点兜底为落地。
- `speed_step`、`low_speed_start`、`speed_drop` 需要连续可见窗口，避免短暂丢球后的缺失点污染速度，制造假落地。
- `speed_step`、`low_speed_start`、`speed_drop` 和 `trajectory_end` 会忽略画面顶部高空区；球仍在画面上半部且最近一段窗口呈向上运动时，也不会把低速点误判成落地。
- 普通可见性丢失会记录为 `out_of_frame` / `visibility_drop_tracking_lost`，不再直接判成落地。
- 如果非顶部/非边缘的 tracking lost 持续达到 `18` 个处理样本，且丢失前速度不超过 `120 px/sample`，会延迟确认 `landing` / `tracking_lost_rally_end`，用于处理真实死球后 TrackNet 长时间找不到球的情况。
- 回合结束后，后续帧、后续击球和后续 BST 预测不再计入当前 accumulator，避免把下一回合混入已结束回合。

### 击球统计

`hit` 事件和 BST 预测会按 `(frame_id, timestamp_ms)` 合并，避免同一帧重复计数：

- 轨迹事件提供 `event_confidence` 和可选的 `ball_court_xy`。
- BST 预测提供击球类型、球员侧、置信度和可选的 `hit_court_xy`。
- 如果 BST 没有提供击球点，会尝试用该球员最近的场地投影点估计。

球员侧解析规则：

- `Top_*` 或 `top_*` -> `top`
- `Bottom_*` 或 `bottom_*` -> `bottom`
- 无侧别时，优先从 `pred_name` 解析，显示名只作为补充。

击球区域按该球员半场深度三等分：

- `front`
- `mid`
- `back`

### 球员运动统计

`update_frame(...)` 使用场地坐标中的球员点统计运动：

- 累计跑动距离。
- 平均速度、最大速度。
- 启动次数、急停次数。
- 高强度移动次数。
- 最大连续移动距离。
- 前后/左右移动比例。
- 平均站位深度。

默认过滤与阈值：

- `min_step_cm = 2.0`：小于该步长视为抖动，不累计距离。
- `max_step_cm = 180.0`：超过该步长视为跳变，重置上一点。
- `start_speed_mps = 1.20`：从静止进入移动的阈值。
- `stop_speed_mps = 0.35`：从移动进入停止的阈值。
- `high_intensity_speed_mps = 3.00`：高强度移动阈值。
- `passive_hit_speed_mps = 0.75`：击球时球员速度低于该值则记为被动/低移动击球。

## 与 MonoTrack 的差异

| 项目 | MonoTrack | WFBARNet 当前 |
| --- | --- | --- |
| 回合输入 | 默认已切好的 `rally_video/*.mp4` | 默认当前视频/片段作为一个统计单元 |
| 自动切分多回合 | 主流程没有；有人工 CSV 切分脚本 | 没有 |
| 回合开始依据 | 人工切片边界 | 稳定球飞行段或第一条 `hit` |
| 回合结束依据 | 数据集切片边界；内部主要处理 hit 序列 | 可信 `landing` 轨迹事件；长 tracking lost 可延迟确认结束；`out_of_frame` 只计数 |
| 击球检测 | HitNet + 后处理/DP；另有局部极值启发式 | `RealtimeTrajectoryEventDetector` 的轨迹反转规则作为主 hit，BST 只做动作分类 |
| 击球间隔/交替约束 | `MLHitDetector.dp_postprocessing` 中有双方交替和最小间隔约束 | 主 hit 当前没有强制双方交替；只做事件冷却和轨迹分数过滤 |
| 回合统计 | 主要服务 3D 重建和分析 | 明确输出 UI/导出的 `rally_record` |

## 当前限制

- 当前不是裁判级回合判定，不能保证 `landing` 就一定是死球，也不能判断是否界内、是否犯规、是否重发。
- 如果一个输入视频包含多个回合，当前 `RallyStatsAccumulator` 会冻结第一个已结束回合，但不会自动创建下一条 rally record。
- 回合结束后，后续帧不会继续累计到当前 accumulator。
- `rally_state` 依赖轨迹事件质量。误检落地会提前显示 `回合结束`，漏检落地会保持 `回合中`。
- 当前没有发球检测、比分牌 OCR、镜头切换检测或长静止片段检测。

## 相关文档

- `assets/docs/hit_detection_pipeline.md`：主击球、落点和出画事件检测。
- `assets/docs/current_prediction_pipeline.md`：当前 PyQt 主预测链路。
- `assets/docs/data_tab_plan.md`：数据页展示和统计字段规划。
- `src/postprocess/rally_stats.py`：当前回合统计实现。
- `tests/test_rally_stats.py`：当前回合统计行为测试。
