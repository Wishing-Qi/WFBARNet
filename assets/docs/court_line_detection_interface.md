# 球场线检测模块接口

本文档记录当前球场线检测模块的统一接口、原 OpenCV 默认后端、MonoTrack 可选后端参数，以及 PyQt 服务层调用方式。

## 模块位置

| 模块 | 作用 |
| --- | --- |
| `src/court/court_line_detector.py` | 统一接口层，提供后端工厂和单帧快捷调用。 |
| `src/court/opencv_court_detector.py` | 原 OpenCV 球场线检测后端，默认用于 PyQt。 |
| `src/court/monotrack_court_detector.py` | MonoTrack 风格球场线检测后端，保留为可选后端。 |
| `apps/pyqt6/services/court_detection_service.py` | PyQt 异步检测服务，内部通过统一接口创建检测器。 |

## 快速调用

```python
from src.court import create_court_line_detector, predict_court_lines

detector = create_court_line_detector()  # 默认 backend="opencv"
prediction = detector.predict(frame, frame_id=0, timestamp_ms=0, force=True)

# 单帧快捷调用
prediction = predict_court_lines(frame, frame_id=0, timestamp_ms=0)
```

## 统一接口

### `CourtLineBackend`

```python
CourtLineBackend = Literal["monotrack", "opencv"]
```

| 值 | 后端 |
| --- | --- |
| `opencv` | 项目原有 OpenCV 球场线检测后端，当前默认后端。 |
| `monotrack` | MonoTrack 风格传统 CV 球场线检测，可显式选择。 |

### `CourtLineConfig`

```python
CourtLineConfig = MonoTrackCourtLineConfig | OpenCVCourtLineConfig
```

`config` 必须和 `backend` 匹配：

| backend | config 类型 |
| --- | --- |
| `monotrack` | `MonoTrackCourtLineConfig` |
| `opencv` | `OpenCVCourtLineConfig` |

类型不匹配时，`create_court_line_detector(...)` 会抛出 `TypeError`。

### `CourtLineDetector`

协议接口，所有球场线检测后端都需要实现：

```python
class CourtLineDetector(Protocol):
    def reset(self) -> None: ...

    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool = False,
    ) -> CourtLinePrediction: ...

    def latest_prediction(self) -> CourtLinePrediction | None: ...
```

| 方法 | 参数 | 返回 | 说明 |
| --- | --- | --- | --- |
| `reset()` | 无 | `None` | 清空内部跟踪状态和最近一次预测。 |
| `predict(...)` | `frame`, `frame_id`, `timestamp_ms`, `force` | `CourtLinePrediction` | 对当前帧执行或复用球场线检测。 |
| `latest_prediction()` | 无 | `CourtLinePrediction | None` | 返回最近一次预测结果。 |

#### `predict(...)` 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `frame` | `np.ndarray` | 必填 | BGR 图像帧，通常来自 OpenCV。要求至少二维，彩色帧通常为 `H x W x 3`。 |
| `frame_id` | `int` | 必填 | 当前帧编号，用于输出记录和重检测节流。 |
| `timestamp_ms` | `int` | 必填 | 当前帧时间戳，单位毫秒。内部会归一化为非负整数。 |
| `force` | `bool` | `False` | 是否强制本帧重新检测。为 `False` 时，检测器会按 `redetect_interval` 和当前状态决定是否复用结果。 |

### `create_court_line_detector(...)`

```python
def create_court_line_detector(
    backend: CourtLineBackend = "opencv",
    *,
    config: CourtLineConfig | None = None,
) -> CourtLineDetector
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `backend` | `CourtLineBackend` | `"opencv"` | 选择检测后端。PyQt 当前默认使用 `opencv`。 |
| `config` | `CourtLineConfig | None` | `None` | 后端配置。为 `None` 时使用该后端默认配置。 |

### `predict_court_lines(...)`

```python
def predict_court_lines(
    frame: np.ndarray,
    *,
    frame_id: int = 0,
    timestamp_ms: int = 0,
    detector: CourtLineDetector | None = None,
    backend: CourtLineBackend = "opencv",
    config: CourtLineConfig | None = None,
    force: bool = True,
) -> CourtLinePrediction
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `frame` | `np.ndarray` | 必填 | BGR 图像帧。 |
| `frame_id` | `int` | `0` | 帧编号。 |
| `timestamp_ms` | `int` | `0` | 时间戳，单位毫秒。 |
| `detector` | `CourtLineDetector | None` | `None` | 可传入已有检测器以复用状态；为 `None` 时内部创建新检测器。 |
| `backend` | `CourtLineBackend` | `"opencv"` | 当 `detector is None` 时使用的后端。 |
| `config` | `CourtLineConfig | None` | `None` | 当 `detector is None` 时使用的配置。 |
| `force` | `bool` | `True` | 单帧快捷调用默认强制检测。 |

## MonoTrack 后端

### `MonoTrackCourtLineDetector`

```python
detector = MonoTrackCourtLineDetector(config=None)
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `config` | `MonoTrackCourtLineConfig | None` | `None` | MonoTrack 后端配置。为 `None` 时使用默认参数。 |

### 算法流程

MonoTrack 后端是纯 Python/OpenCV 实现，移植的是 MonoTrack 的传统 CV 思路：

1. 对帧做灰度亮度检测，寻找局部亮脊线像素。
2. 使用结构张量过滤非线状亮点。
3. 使用 `cv2.HoughLinesP` 提取候选直线段。
4. 先做三方向角度聚类，优先走三方向模板拟合。
5. 若三方向拟合失败或置信度过低，再回退到两方向模板枚举。
6. 用透视变换拟合标准场地模板。
7. 选择与白线二值图重合度最高的模型。
8. 通过统一 `CourtLinePrediction` 输出角点、单应矩阵和投影场线。

### `MonoTrackCourtLineConfig`

```python
@dataclass(slots=True)
class MonoTrackCourtLineConfig:
    redetect_interval: float = 4.0
    detect_max_width: int = 960
    luminance_threshold: int = 80
    diff_threshold: int = 20
    ridge_offset_px: int = 4
    gradient_kernel_size: int = 3
    structure_kernel_size: int = 21
    hough_threshold: int = 50
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10
    angle_bin_deg: float = 5.0
    angle_tol_deg: float = 16.0
    min_angle_separation_deg: float = 25.0
    merge_rho_px: float = 16.0
    max_lines_per_family: int = 3
    model_sample_step_px: float = 8.0
    model_sample_radius_px: int = 2
    point_scheme: str = "auto"
    refine_homography: bool = True
    snap_search_px: float = 18.0
    snap_response_threshold: float = 0.18
    max_refine_corner_shift_ratio: float = 0.045
    green_side_offset_px: float = 14.0
    min_outer_width_ratio: float = 0.08
    min_outer_depth_ratio: float = 0.08
    min_outer_width_depth_ratio: float = 0.18
    max_outer_width_depth_ratio: float = 5.5
    max_transverse_angle_deg: float = 35.0
    reliable_conf: float = 0.68
    medium_conf: float = 0.48
    smooth_alpha_reliable: float = 0.45
    smooth_alpha_medium: float = 0.20
    jump_ratio_hard: float = 0.18
```

#### 检测调度参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `redetect_interval` | `float` | `4.0` | 非强制模式下，两次自动重检测之间的最短间隔，单位秒。 |
| `detect_max_width` | `int` | `960` | 检测前最大缩放宽度。原帧宽度超过该值时会按比例缩小，检测结果再映射回原图。 |

#### 白线像素参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `luminance_threshold` | `int` | `80` | 候选白线像素最低亮度阈值。 |
| `diff_threshold` | `int` | `20` | 当前像素相对左右或上下邻域的最小亮度差。 |
| `ridge_offset_px` | `int` | `4` | 比较局部亮脊时，向左右/上下采样的像素偏移。对应 MonoTrack 原实现中的 `t`。 |
| `gradient_kernel_size` | `int` | `3` | Sobel 梯度核大小，用于结构张量计算。偶数会自动调成奇数。 |
| `structure_kernel_size` | `int` | `21` | 结构张量积分窗口大小。越大越偏向保留长线状结构。 |

#### Hough 直线参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `hough_threshold` | `int` | `50` | `cv2.HoughLinesP` 的投票阈值。 |
| `hough_min_line_length` | `int` | `50` | Hough 线段最小长度，单位像素。 |
| `hough_max_line_gap` | `int` | `10` | Hough 线段最大断裂连接距离，单位像素。 |

#### 线族聚类参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `angle_bin_deg` | `float` | `5.0` | 方向直方图角度分箱大小。 |
| `angle_tol_deg` | `float` | `16.0` | 将线段归入某个方向族时允许的角度偏差。 |
| `min_angle_separation_deg` | `float` | `25.0` | 两个主要方向族之间的最小角度差。 |
| `merge_rho_px` | `float` | `16.0` | 同方向线按法线距离合并时的距离阈值，单位像素。 |
| `max_lines_per_family` | `int` | `3` | 每个方向族最多参与模板枚举的合并线数量。值越大越慢，但候选更全。 |

#### 模板评分参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model_sample_step_px` | `float` | `8.0` | 沿投影模板线采样的步长，单位像素。越小越精细但越慢。 |
| `model_sample_radius_px` | `int` | `2` | 采样点周围用于判断白线命中的半径，单位像素。 |

#### 三方向拟合参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `point_scheme` | `str` | `"auto"` | 关键点方案，自动在 6 点/8 点之间择优。 |
| `refine_homography` | `bool` | `True` | 是否在拟合后用白线采样点进一步细化单应矩阵。 |
| `snap_search_px` | `float` | `18.0` | 白线细化时沿法线方向搜索的最大像素范围。 |
| `snap_response_threshold` | `float` | `0.18` | 细化采样点接受为有效命中的最低响应。 |
| `max_refine_corner_shift_ratio` | `float` | `0.045` | 细化后角点位移占图像对角线的最大允许比例。 |
| `green_side_offset_px` | `float` | `14.0` | 外框两侧绿色支撑采样偏移，MonoTrack 当前默认保留该参数位。 |

#### 几何约束参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `min_outer_width_ratio` | `float` | `0.08` | 外框最大宽度相对图像对角线的最小比例。 |
| `min_outer_depth_ratio` | `float` | `0.08` | 外框最大深度相对图像对角线的最小比例。 |
| `min_outer_width_depth_ratio` | `float` | `0.18` | 外框宽深比下限。 |
| `max_outer_width_depth_ratio` | `float` | `5.5` | 外框宽深比上限。 |
| `max_transverse_angle_deg` | `float` | `35.0` | 判断横向场线族时允许偏离水平线的最大角度。 |

#### 跟踪与平滑参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `reliable_conf` | `float` | `0.68` | 候选置信度达到该值时作为可靠更新。 |
| `medium_conf` | `float` | `0.48` | 候选置信度达到该值但低于可靠阈值时，只有已有当前结果才会做平滑更新。 |
| `smooth_alpha_reliable` | `float` | `0.45` | 可靠更新时，新候选角点的融合权重。 |
| `smooth_alpha_medium` | `float` | `0.20` | 中等置信度更新时，新候选角点的融合权重。 |
| `jump_ratio_hard` | `float` | `0.18` | 与上一帧场地角点平均跳变过大时的惩罚阈值，相对图像对角线。 |

## OpenCV 后端参数

原 OpenCV 后端是当前默认后端，也可显式传入 `backend="opencv"` 和自定义参数：

```python
from src.court import OpenCVCourtLineConfig, create_court_line_detector

detector = create_court_line_detector(
    backend="opencv",
    config=OpenCVCourtLineConfig(redetect_interval=2.0),
)
```

`OpenCVCourtLineConfig` 参数如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `redetect_interval` | `4.0` | 自动重检测间隔，单位秒。 |
| `detect_max_width` | `960` | 检测最大宽度。 |
| `white_s_max` | `130` | HSV 白线饱和度上限。 |
| `white_v_min` | `120` | HSV 白线亮度下限。 |
| `white_chroma_max` | `96` | Lab 色度距离上限。 |
| `line_response_percentile` | `91.0` | 白线响应自适应分位阈值。 |
| `line_response_min` | `72` | 白线响应最低阈值。 |
| `line_local_bg_ksize` | `31` | 局部背景估计核大小。 |
| `use_green_roi` | `True` | 是否使用绿色场地区域约束白线。 |
| `green_h_min` | `30` | HSV 绿色 H 下限。 |
| `green_h_max` | `100` | HSV 绿色 H 上限。 |
| `green_s_min` | `70` | HSV 绿色 S 下限。 |
| `green_v_min` | `35` | HSV 绿色 V 下限。 |
| `white_green_pair_offset_px` | `8` | 检查白线两侧绿色支撑的采样偏移。 |
| `keep_all_green_rois` | `False` | 是否保留所有绿色连通区域。 |
| `hough_threshold` | `45` | Hough 投票阈值。 |
| `min_line_length_ratio` | `0.055` | Hough 最小线段长度相对图像对角线比例。 |
| `max_line_gap_ratio` | `0.025` | Hough 最大断裂距离相对图像对角线比例。 |
| `angle_bin_deg` | `5.0` | 方向直方图角度分箱。 |
| `angle_tol_deg` | `16.0` | 线段归属方向族的角度容差。 |
| `min_angle_separation_deg` | `25.0` | 两个方向族最小角度差。 |
| `merge_rho_px` | `18.0` | 同方向线合并距离。 |
| `max_lines_per_family` | `3` | 每个方向族最多参与枚举的合并线数量。 |
| `point_scheme` | `"auto"` | 关键点方案，支持自动选择。 |
| `refine_homography` | `True` | 是否用白线采样点细化单应矩阵。 |
| `snap_search_px` | `18.0` | 细化时沿法线搜索白线的半径。 |
| `snap_response_threshold` | `0.18` | 细化采样点最低响应阈值。 |
| `max_refine_corner_shift_ratio` | `0.045` | 单应细化后角点最大平均偏移比例。 |
| `green_side_offset_px` | `14.0` | 外框两侧绿色支撑采样偏移。 |
| `min_outer_width_ratio` | `0.08` | 外框宽度最小比例。 |
| `min_outer_depth_ratio` | `0.08` | 外框深度最小比例。 |
| `min_outer_width_depth_ratio` | `0.18` | 外框宽深比下限。 |
| `max_outer_width_depth_ratio` | `5.5` | 外框宽深比上限。 |
| `max_transverse_angle_deg` | `35.0` | 横向线族最大偏离角。 |
| `reliable_conf` | `0.75` | 可靠更新置信度阈值。 |
| `medium_conf` | `0.55` | 中等更新置信度阈值。 |
| `smooth_alpha_reliable` | `0.45` | 可靠更新融合权重。 |
| `smooth_alpha_medium` | `0.20` | 中等更新融合权重。 |
| `jump_ratio_hard` | `0.18` | 时序跳变惩罚阈值。 |
| `mask_alpha` | `0.14` | 绘制覆盖层的场地填充透明度。 |
| `line_thickness` | `3` | 绘制场线粗细。 |
| `point_radius` | `5` | 绘制关键点半径。 |
| `show_labels` | `False` | 是否绘制关键点标签。 |
| `draw_debug_lines` | `False` | 是否绘制调试线族。 |

## 输出结构

### `CourtLinePrediction`

所有后端统一输出该对象，并可通过 `to_dict()` 转为 UI/日志使用的字典。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `frame_id` | `int` | 当前帧编号。 |
| `timestamp_ms` | `int` | 当前帧时间戳，单位毫秒。 |
| `source_size` | `tuple[int, int]` | 原始帧尺寸，格式为 `(width, height)`。 |
| `valid` | `bool` | 当前是否有可用球场检测结果。 |
| `attempted` | `bool` | 本帧是否尝试重新检测。 |
| `updated` | `bool` | 本帧是否用候选结果更新了当前状态。 |
| `update_type` | `str` | 跟踪状态更新类型，例如 `reliable update`、`medium smooth`、`rejected`。 |
| `status` | `str` | 面向 UI 的状态文本。 |
| `confidence` | `float` | 当前结果置信度，范围通常为 `0..1`。 |
| `candidate_confidence` | `float | None` | 本次候选置信度；本帧未检测时可能为 `None`。 |
| `reason` | `str` | 当前结果或候选的评分原因。 |
| `scheme` | `str` | 关键点/后端方案。MonoTrack 后端当前使用 `"monotrack"`。 |
| `corners` | `list[list[float]]` | 图像中的外框四角，顺序为 `top-left, top-right, bottom-right, bottom-left`。 |
| `keypoints` | `list[dict]` | 关键点列表，每项包含 `name` 和 `point`。 |
| `court_to_image_h` | `list[list[float]]` | 标准场地坐标到图像坐标的 3x3 单应矩阵。 |
| `image_to_court_h` | `list[list[float]]` | 图像坐标到标准场地坐标的 3x3 单应矩阵。 |
| `projected_lines` | `dict[str, list[list[float]]]` | 投影到图像上的标准场线。 |
| `metrics` | `dict[str, Any]` | 检测诊断指标，例如线数量、模板支撑、评分组件。 |
| `detect_ms` | `float` | 本帧检测耗时，单位毫秒；复用结果时为 `0.0`。 |
| `rejected_count` | `int` | 连续候选被拒绝次数。 |

### `projected_lines` 常见键

| 键 | 含义 |
| --- | --- |
| `doubles_outer` | 双打外框四边形。 |
| `singles_left_sideline` | 左单打边线。 |
| `singles_right_sideline` | 右单打边线。 |
| `top_short_service` | 上半场前发球线。 |
| `bottom_short_service` | 下半场前发球线。 |
| `top_doubles_long_service` | 上半场双打后发球线。 |
| `bottom_doubles_long_service` | 下半场双打后发球线。 |
| `top_center_service` | 上半场中线。 |
| `bottom_center_service` | 下半场中线。 |

## PyQt 服务接口

### `CourtDetectionService`

```python
service = CourtDetectionService(
    config=None,
    backend="opencv",
    submit_interval_s=0.75,
)
service.start()
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `config` | `CourtLineConfig | None` | `None` | 检测后端配置。必须与 `backend` 对应。 |
| `backend` | `CourtLineBackend` | `"opencv"` | PyQt 检测服务使用的后端。 |
| `submit_interval_s` | `float` | `0.75` | 后台线程接受新帧的最短间隔，避免 UI 高频提交导致堆积。 |

### `create_court_detection_service(...)`

```python
def create_court_detection_service(
    config: CourtLineConfig | None = None,
    *,
    backend: CourtLineBackend = "opencv",
) -> CourtDetectionService
```

该函数会创建服务并立即调用 `start()`。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `config` | `CourtLineConfig | None` | `None` | 后端配置。 |
| `backend` | `CourtLineBackend` | `"opencv"` | 服务后端。默认使用原 OpenCV 检测模块。 |

### 服务方法

| 方法 | 参数 | 返回 | 说明 |
| --- | --- | --- | --- |
| `start()` | 无 | `None` | 启动后台检测线程。 |
| `stop()` | 无 | `None` | 请求后台线程停止并等待退出。 |
| `reset()` | 无 | `None` | 重置检测器状态，清空最新预测。 |
| `request_prediction()` | 无 | `None` | 允许下一次 `submit_frame(...)` 被后台线程接受。 |
| `clear_pending()` | 无 | `None` | 清空尚未处理的提交帧。 |
| `submit_frame(frame, frame_id, timestamp_ms)` | `np.ndarray`, `int`, `int` | `bool` | 向后台线程提交一帧。返回 `True` 表示已接受。 |
| `latest_prediction()` | 无 | `CourtLinePrediction | None` | 返回最近一次预测对象。 |
| `latest_prediction_dict()` | 无 | `dict | None` | 返回最近一次预测的字典形式。 |

### PyQt 信号

| 信号 | 参数 | 说明 |
| --- | --- | --- |
| `resultReady` | `object` | 后台完成检测后发出。服务层会发出 `prediction.to_dict()`。 |
| `failed` | `str` | 后台检测异常时发出错误信息。 |

## 使用示例

### PyQt 默认 OpenCV 后端

```python
from apps.pyqt6.services.court_detection_service import create_court_detection_service

court_service = create_court_detection_service()
```

### PyQt 显式选择 MonoTrack 后端

```python
from apps.pyqt6.services.court_detection_service import create_court_detection_service
from src.court import MonoTrackCourtLineConfig

config = MonoTrackCourtLineConfig(
    redetect_interval=2.0,
    hough_threshold=45,
    reliable_conf=0.65,
)
court_service = create_court_detection_service(config, backend="monotrack")
```

### 显式选择原 OpenCV 参数

```python
from apps.pyqt6.services.court_detection_service import create_court_detection_service
from src.court import OpenCVCourtLineConfig

config = OpenCVCourtLineConfig(redetect_interval=4.0)
court_service = create_court_detection_service(config, backend="opencv")
```

## 注意事项

- MonoTrack 后端当前是 Python/OpenCV 移植，不依赖 `D:\Github\monotrack` 中的 C++ 可执行文件。
- 若传入已有 `detector` 给 `predict_court_lines(...)`，`backend` 和 `config` 会被忽略。
- `CourtDetectionService.submit_frame(...)` 只有在先调用 `request_prediction()` 且间隔满足 `submit_interval_s` 时才会接受帧。
- `image_to_court_h` 使用标准羽毛球场坐标，单位与 `opencv_court_homography_core.py` 中模板一致：宽 `610`、长 `1340`。
