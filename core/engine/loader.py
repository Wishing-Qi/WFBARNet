import os
import importlib
import inspect
import logging
from typing import Dict, Type
from interfaces.istream import IStreamPlugin

class PluginLoader:
    """
    插件加载器，负责扫描 plugins/ 目录并动态加载 IStreamPlugin 实例。
    """
    def __init__(self, plugin_dir: str = "plugins/streams"):
        self.plugin_dir = plugin_dir
        self.logger = logging.getLogger("PluginLoader")
        self.available_plugins: Dict[str, Type[IStreamPlugin]] = {}
        self._scan_plugins()

    def _scan_plugins(self):
        """扫描插件目录并加载类"""
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = f"plugins.streams.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, IStreamPlugin) and 
                            obj is not IStreamPlugin):
                            plugin_id = getattr(obj, "PLUGIN_ID", name.lower())
                            self.available_plugins[plugin_id] = obj
                            self.logger.info(f"Loaded stream plugin: {plugin_id}")
                except Exception as e:
                    self.logger.error(f"Failed to load plugin {module_name}: {e}")

    def create_plugin(self, plugin_id: str) -> IStreamPlugin:
        """根据 ID 创建插件实例"""
        if plugin_id in self.available_plugins:
            return self.available_plugins[plugin_id]()
        raise ValueError(f"Plugin {plugin_id} not found.")
