import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import BadmintonMainWindow, AppStatus
from utils.logger import setup_logger
from utils.config_loader import cfg
from core.engine.orchestrator import InferenceEngine

def main():
    # 1. 初始化日志系统
    setup_logger(
        log_level=cfg.get("Logging.level", "INFO"),
        log_dir=cfg.get("Logging.output_dir", "logs")
    )
    
    # 2. 创建桌面应用
    app = QApplication(sys.argv)
    
    # 3. 初始化推理引擎 (单例)
    try:
        engine = InferenceEngine()
    except Exception as e:
        from utils.logger import logger
        logger.error(f"Failed to initialize InferenceEngine: {e}")
        sys.exit(1)
        
    # 4. 创建并显示主窗口
    window = BadmintonMainWindow()
    
    # 初始化状态
    window.update_app_status(AppStatus.IDLE)
    
    window.show()
    
    # 5. 进入主循环
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
