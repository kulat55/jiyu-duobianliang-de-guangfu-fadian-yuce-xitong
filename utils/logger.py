import os
import logging
from datetime import datetime


def init_logger():
    if not os.path.exists("logs"):
        os.makedirs("logs")

    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_path = os.path.join("logs", log_filename)

    logger = logging.getLogger("pv_system")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("系统启动成功，日志文件已创建：%s", log_path)

    return logger