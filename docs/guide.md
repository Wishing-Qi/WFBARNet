### 1. 插件化目录结构更新

通过 `interfaces/` 文件夹定义契约，实现业务逻辑与底层实现的完全隔离。

```text
WFBARNet/
├── interfaces/             # 统一接口定义 (Abstract Base Classes)
│   ├── istream.py          # 流插件标准接口
│   └── idata.py            # 统一数据包格式定义
├── plugins/                # 插件实现层
│   └── streams/            # 流媒体接入插件
│       ├── file_stream.py  # MP4/AVI 插件
│       ├── rtsp_stream.py  # 网络摄像头插件
│       └── webrtc_stream.py# 2026 低延迟推流插件
├── core/                   # 核心调度逻辑 (Orchestrator)
│   ├── engine/             # 插件加载与模型管线调度
│   ├── inference/          # 模型推断包装 (YOLO26/TrackNet/BST)
│   │   └── court_detector.py
│   ├── tracker/            # 球员与球追踪逻辑
│   │   ├── ball_tracker/
│   │   └── human_tracker/
│   ├── classifiers/        # 分类器 (动作识别等)
│   ├── detectors/          # 检测器 (目标检测等)
│   └── base_model.py       # 模型基类
├── ui/                     # PySide6 界面
├── common/                 # 通用数据框架
│   └── packet.py           # 核心数据包 (FramePacket)
├── configs/                # 配置文件
├── data/                   # 数据存储
├── deploy/                 # 部署相关 (模型权重、转换)
├── main.py                 # 程序入口
├── testbench.py            # 测试脚本
├── requirements.txt        # 依赖列表
└── docs/                   # 需求与设计文档
```

---

### 2. 统一数据框架与接口设计

#### 2.1 核心数据包 (`FramePacket`)
这是在系统各个模块间流转的唯一对象，保证了数据传输的标准性。

```python
# common/packet.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    source_id: str
    image: np.ndarray             # 原始图像
    
    # 推理结果占位符 (由 core 模块填充)
    ball_coord: Optional[tuple] = None    # (x, y, conf)
    skeletons: List[Dict] = field(default_factory=list) # [{id: 1, keypoints: [...]}]
    court_info: Dict = field(default_factory=dict)     # 场线坐标
    stroke_action: Optional[str] = None   # 动作类型
    
    metadata: Dict[str, Any] = field(default_factory=dict) # 预留扩展字段
```

#### 2.2 插件接口 (`IStream`)
所有流插件必须实现的接口。

```python
# interfaces/istream.py
from abc import ABC, abstractmethod
from common.packet import FramePacket

class IStreamPlugin(ABC):
    @abstractmethod
    def connect(self, source: str) -> bool:
        """连接视频源"""
        pass

    @abstractmethod
    def read(self) -> FramePacket:
        """读取下一帧数据包"""
        pass

    @abstractmethod
    def release(self):
        """释放资源"""
        pass

    @property
    @abstractmethod
    def is_opened(self) -> bool:
        pass
```

---

### 3. 系统需求文档 (Requirement Specification)

#### 3.1 项目概述
本系统旨在建立一个集“感知-追踪-识别-分析”于一体的羽毛球综合智能系统，通过 2D 视角图像实现 3D 物理层面的比赛复盘与球员技术统计。

#### 3.2 核心功能需求 (Functional Requirements)
1.  **多源接入能力：** 
    *   支持插件式扩展，首期需兼容本地文件、RTSP 及 WebRTC。
    *   具备自动断线重连机制。
2.  **视觉感知精度：**
    *   **球迹追踪：** TrackNet V3 在 4K 画面下的漏检率需低于 5%（除严重遮挡外）。
    *   **姿态识别：** YOLO26-Pose 需在 40 系显卡实现 > 60fps 的实时推断。
    *   **击球识别：** BST 模型需识别至少 8 种基础击球动作，准确率 > 85%。
3.  **球员追踪与属性绑定：**
    *   支持多球员（双打）追踪，ID 切换率 (ID Switch) 低于 3 次/局。
4.  **UI 交互：**
    *   提供实时视频叠加渲染。
    *   可视化球员热力图及击球分布图。

#### 3.3 非功能性需求 (Non-Functional Requirements)
1.  **硬件自适应：** 
    *   **高性能模式：** 检测到 RTX 40 系显卡时，自动启用 TensorRT 10.x 加速。
    *   **兼容模式：** 在无 GPU 环境下，自动切换为 ONNX Runtime (CPU) 推理，此时允许降低采样率（如从 60fps 降至 15fps）。
2.  **可扩展性：** 
    *   流媒体、模型算法均需支持插件化热插拔。
    *   逻辑处理模块与 UI 显示模块必须位于不同进程/线程，避免阻塞。
3.  **时间戳对齐：** 系统需处理各模型推理耗时差异，确保 TrackNet 轨迹与 YOLO 骨架在同一 `FramePacket` 中对齐。

#### 3.4 运行环境要求
*   **OS:** Windows 10/11 或 Ubuntu 24.04+
*   **Language:** Python 3.10.20
*   **Driver:** NVIDIA Driver >= 555.xx (for CUDA 12.x)

---

### 4. 自我审查 (Self-Critique)

*   **数据积压风险：** 插件化读取可能导致生产者（Stream）速度远快于消费者（Inference）。**修正：** 必须在插件接口中强制要求实现缓冲管理，当缓冲区满时，旧帧自动覆盖或丢弃。
*   **插件发现机制：** 目录结构中虽然预留了 `plugins/`，但需要一个自动加载机制。**建议：** 使用 Python 的 `importlib` 动态加载该文件夹下的所有类。
。
