# TrackNet Demo

TrackNet 实时推理演示程序 - 直接对视频进行乒乓球轨迹跟踪。

## 功能特性

- TrackNet 实时乒乓球检测与跟踪
- 检测结果可视化（红色圆点标记球位置）
- 实时显示 FPS 和跟踪状态
- **自动 GPU 加速**（支持 CUDA GPU）
- 复用车项目核心推理代码

## 前提条件

确保已下载 TrackNet 模型权重文件：
- 权重文件路径: `assets/weights/track/model_best.pt`
- 如需下载，请参考: [WFBARNet 下载指南](https://github.com/your-repo/WFBARNet#download-models)

## 安装依赖

```bash
pip install opencv-python torch numpy
```

## 运行

```bash
cd tools/demo
python tracknet_demo.py
```

## 使用方法

1. 程序会自动加载 `videos/test3.mp4` 视频
2. 自动检测并使用可用的 GPU（优先 CUDA）
3. 实时显示跟踪结果
4. 按 `q` 或 `ESC` 退出

## 技术细节

- 使用 TrackNet V3 模型进行球轨迹跟踪
- 直接复用 `src.models.track_branch.TrackBranch` 后端代码
- 自动 GPU 加速检测（CUDA）
- 输入尺寸: 512x288
- 检测阈值: 0.35

## 显示信息

| 信息 | 说明 |
|------|------|
| FPS | 实时帧率 |
| Frame | 当前帧/总帧数 |
| Track | 球位置 (x, y) 和置信度 |
| 红色圆点 | 检测到的球位置 |
| 黄色圆圈 | 球周围的高亮框 |
