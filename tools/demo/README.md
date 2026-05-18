# Demo 脚本

这个目录存放一些用于本地实验和模型排查的独立 demo。

## 只跑 YOLO 原始推理

`run_yolo_only_infer.py` 用于排查 YOLO 模型本身的输出。它只调用
`ultralytics.YOLO.predict()` 和 `result.plot()`，不会执行球场线后处理、白线吸附、
单应性估计或标准球场模板投影。适合判断问题到底来自 YOLO 原始检测，还是来自后续
球场几何拟合流程。

脚本顶部有 `RUN_CONFIG` 配置块，常用参数已经写在文件里。默认 `source` 使用
`videos/MVI_0212.MP4` 的绝对路径，会自动查找 ShuttleCourt 权重，并打开实时预览窗口。
默认不会输出视频文件：

```powershell
python tools/demo/run_yolo_only_infer.py
```

如果临时需要覆盖文件内配置，也可以继续传命令行参数：

```powershell
python tools/demo/run_yolo_only_infer.py `
  --source path/to/frame_or_video.mp4 `
  --weights assets/weights/ShuttleCourtNet/ShuttleCourt.pt `
  --device cpu `
  --imgsz 416 `
  --max-frames 20 `
  --save-frames
```

输出默认写入 `outputs/yolo_only_demo/`：

- `*_yolo_only.jpg`：图片输入的 YOLO 原始可视化。
- `frames/*_yolo_only.jpg`：仅在 `RUN_CONFIG["save_frames"] = True` 或使用 `--save-frames` 时保存抽帧可视化。
- `*_yolo_only.json`：每帧 bbox、mask polygon、keypoints 和置信度等原始输出。

如果需要输出 mp4，可以显式把 `RUN_CONFIG["write_video"]` 改为 `True`，或运行时追加
`--write-video`。

## 球场关键点检测

`run_court_keypoints_yolo.py` 用于检查 `assets/weights/ShuttleCourtNet/ShuttleCourt.pt`
的原始 ShuttleCourt YOLO pose 输出。它主要用于模型诊断：脚本会绘制检测到的球场
bbox、8 个关键点编号、每个点的置信度、可选关键点分组，并将原始检测结果导出为 JSON。

预期模型输出：

- 1 个 `court` 目标。
- 8 个关键点。
- 可视化默认分组：球场线点 `0,1,2,3`，网线点 `4,5,6,7`。
- 默认原始场面遮罩映射：`0,2,3,1`。
- 默认推理尺寸为 `1280`，与内置权重元信息中的训练尺寸一致。

运行内置视频 `videos/MVI_0212.MP4`：

```powershell
python tools/demo/run_court_keypoints_yolo.py
```

在自定义视频上运行：

```powershell
python tools/demo/run_court_keypoints_yolo.py `
  --source path/to/video.mp4 `
  --imgsz 1280 `
  --stride 30 `
  --max-frames 300 `
  --log-every 100 `
  --device cpu
```

输出默认写入 `outputs/shuttlecourt_demo/`：

- `*_shuttlecourt.json`：原始检测结果。
- `*_shuttlecourt.jpg`：图片输入的可视化结果。
- `*_shuttlecourt.mp4`：视频输入的可视化结果，除非使用 `--no-video`。
- `frames/*.jpg`：使用 `--save-frames` 时保存的已处理视频帧。

快速检查原始模型输出时，可以抽样处理少量帧并保存为图片：

```powershell
python tools/demo/run_court_keypoints_yolo.py `
  --source videos/MVI_0212.MP4 `
  --device cpu `
  --max-frames 8 `
  --stride 120 `
  --no-video `
  --save-frames
```

如果模型的关键点顺序不同，可以显式传入索引映射：

```powershell
--surface-indices 0,2,3,1 --court-indices 0,1,2,3 --net-indices 4,5,6,7
```

半透明遮罩使用 `--surface-indices`，不是直接使用前 4 个原始关键点。可以用下面的参数
关闭遮罩、调整透明度、隐藏标签，或显示所有检测到的球场候选：

```powershell
--no-mask
--mask-alpha 0.12
--hide-labels
--draw-all
```

建议先用这个脚本排查原始关键点，再调试 homography 叠加层。如果带编号的原始关键点已经
错了，问题通常在模型、数据域或训练标注质量；如果原始关键点看起来正确，但投影球场错误，
问题通常在关键点索引映射或几何层。

## ShuttleCourt 实时 GPU

`run_shuttlecourt_realtime_gpu.py` 默认使用 GPU 运行实时 ShuttleCourt 推理。

使用默认摄像头和 GPU 0：

```powershell
python tools/demo/run_shuttlecourt_realtime_gpu.py --source 0 --device 0
```

对视频文件做实时播放式推理：

```powershell
python tools/demo/run_shuttlecourt_realtime_gpu.py `
  --source videos/MVI_0211.MP4 `
  --device 0 `
  --surface-indices 0,2,3,1 `
  --log-every 30
```

按 `q` 或 `Esc` 退出显示窗口。可以追加
`--save-video outputs/shuttlecourt_realtime.mp4` 保存可视化视频。

## 其他运行时 Demo

- `run_pose_only.py`
- `run_track_only.py`
- `run_tracknet_realtime.py`
- `run_shuttlecourt_realtime_gpu.py`
- `run_unified_infer.py`
- `tracknet_demo.py`
