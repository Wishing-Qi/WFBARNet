import yaml
import os
from pathlib import Path
from utils.logger import logger

class Config:
    """
    配置加载器单例模式。
    支持 YAML 加载、路径校验与动态覆盖。
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._config_data = {}
            cls._instance._load_all_configs()
        return cls._instance

    def _load_all_configs(self):
        config_dir = Path("configs")
        config_files = ["model_cfg.yml", "pipeline_cfg.yml", "court_cfg.yml"]
        
        for file_name in config_files:
            file_path = config_dir / file_name
            if not file_path.exists():
                logger.error(f"Config file not found: {file_path}")
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data:
                    self._config_data.update(data)
        
        # 自动校验权重路径
        self._validate_weights()

    def _validate_weights(self):
        """检查模型权重文件是否存在"""
        weight_keys = [
            ("YOLO", "weight_path"),
            ("TrackNetV3", "weight_path"),
            ("BST", "weight_path")
        ]
        
        for section, key in weight_keys:
            if section in self._config_data and key in self._config_data[section]:
                path = self._config_data[section][key]
                if not os.path.exists(path):
                    logger.warning(f"Weight file for {section} not found at: {path}")

    def get(self, key_path: str, default=None):
        """
        根据路径获取配置值。支持 "Section.Key" 格式。
        """
        parts = key_path.split('.')
        val = self._config_data
        try:
            for part in parts:
                val = val[part]
            return val
        except (KeyError, TypeError):
            return default

    def override(self, key_path: str, value):
        """
        动态覆盖配置项（用于 CLI 参数注入）。
        """
        parts = key_path.split('.')
        ref = self._config_data
        for part in parts[:-1]:
            if part not in ref:
                ref[part] = {}
            ref = ref[part]
        ref[parts[-1]] = value
        logger.info(f"Config override: {key_path} = {value}")

# 全局配置对象
cfg = Config()
