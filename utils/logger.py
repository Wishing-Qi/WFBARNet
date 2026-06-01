import sys
import time
import functools
import threading
from pathlib import Path
from loguru import logger
import torch

# --- 日志系统配置 ---

def setup_logger(log_level: str = "INFO", log_dir: str = "logs"):
    """
    配置全局日志管理。
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # 移除默认配置
    logger.remove()

    # 控制台输出 (带颜色)
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> [F:{extra[frame_id]}] - <level>{message}</level>",
        enqueue=True # 保证线程安全
    )

    # 文件输出 (按天滚动)
    logger.add(
        log_path / "runtime_{time}.log",
        rotation="00:00",
        retention="10 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} [F:{extra[frame_id]}] - {message}",
        compression="zip",
        enqueue=True
    )

# 默认绑定一个 frame_id 为 0，防止没有绑定时报错
logger = logger.bind(frame_id=0)

# --- 性能监控工具 ---

class PerformanceTimer:
    """
    性能计时器，支持上下文管理器和统计功能。
    """
    _stats = {}
    _lock = threading.Lock()

    def __init__(self, name: str, frame_id: int = 0):
        self.name = name
        self.frame_id = frame_id
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000  # 毫秒
        with self._lock:
            if self.name not in self._stats:
                self._stats[self.name] = []
            self._stats[self.name].append(elapsed)
            
            # 限制统计队列长度，只保留最近 100 帧进行平均
            if len(self._stats[self.name]) > 100:
                self._stats[self.name].pop(0)

    @classmethod
    def get_average(cls, name: str) -> float:
        with cls._lock:
            vals = cls._stats.get(name, [])
            return sum(vals) / len(vals) if vals else 0.0

def time_it(name: str):
    """
    装饰器版计时器。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                # 装饰器版通常用于内部组件，不一定有 frame_id 上下文
                # 这里可以扩展从 args 中提取
                pass 
        return wrapper
    return decorator

# --- 异常捕获与 GPU 状态监控 ---

def catch_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            logger.critical("CUDA Out of Memory (OOM) detected!")
            _log_gpu_status()
            sys.exit(1)
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}: {e}")
            _log_gpu_status()
            raise e
    return wrapper

def _log_gpu_status():
    """记录当前 GPU 显存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / (1024**2)
            mem_res = torch.cuda.memory_reserved(i) / (1024**2)
            logger.error(f"GPU [{i}] - Allocated: {mem_alloc:.2f}MB, Reserved: {mem_res:.2f}MB")
