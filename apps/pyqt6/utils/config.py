from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    app_name: str = "羽毛球分析系统"
    default_window_title: str = "羽毛球分析系统"
    default_video_path: str = "videos/sample.mp4"


APP_CONFIG = AppConfig()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_default_video_path() -> Path:
    return get_project_root() / APP_CONFIG.default_video_path
