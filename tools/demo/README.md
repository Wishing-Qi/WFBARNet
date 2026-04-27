# Demo Scripts

This folder contains small standalone demos for local experiments.

## Court Keypoint Detection

`run_court_keypoints_yolo.py` detects ShuttleCourt keypoints with `assets/weights/ShuttleCourtNet/ShuttleCourt.pt`.

Expected model output:

- one court object
- eight keypoints
- default model keypoint groups: court-line points `0,1,2,3`, net points `4,5,6,7`
- default court mask mapping: `0,2,3,1` as top-left, top-right, bottom-right, bottom-left

Run the small bundled video demo on `videos/MVI_0211.MP4`:

```powershell
python tools/demo/run_court_keypoints_yolo.py
```

Run on a custom video:

```powershell
python tools/demo/run_court_keypoints_yolo.py `
  --source path/to/video.mp4 `
  --max-frames 300 `
  --log-every 100 `
  --device cpu
```

Outputs are written to `outputs/shuttlecourt_demo/`:

- `*_shuttlecourt.json`
- `*_shuttlecourt.jpg` for images
- `*_shuttlecourt.mp4` for videos unless `--no-video` is used

If your model uses a different keypoint order, pass:

```powershell
--surface-indices 0,2,3,1 --court-indices 0,1,2,3 --net-indices 4,5,6,7
```

The translucent mask uses `--surface-indices`, not the raw first four keypoints. To disable
the mask, tune opacity, or draw raw model point numbers for calibration:

```powershell
--no-mask
--mask-alpha 0.12
--draw-raw-points
```

## ShuttleCourt Realtime GPU

`run_shuttlecourt_realtime_gpu.py` runs realtime ShuttleCourt inference with GPU by default.

Run with the default camera on GPU 0:

```powershell
python tools/demo/run_shuttlecourt_realtime_gpu.py --source 0 --device 0
```

Run realtime playback on a video file:

```powershell
python tools/demo/run_shuttlecourt_realtime_gpu.py `
  --source videos/MVI_0211.MP4 `
  --device 0 `
  --surface-indices 0,2,3,1 `
  --log-every 30
```

Press `q` or `Esc` to quit the display window. Add `--save-video outputs/shuttlecourt_realtime.mp4`
to record the visualization.

## Existing Runtime Demos

- `run_pose_only.py`
- `run_track_only.py`
- `run_tracknet_realtime.py`
- `run_shuttlecourt_realtime_gpu.py`
- `run_unified_infer.py`
- `tracknet_demo.py`
