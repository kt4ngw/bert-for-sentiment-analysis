import torch

def get_device():
    """返回可用设备 (CPU / GPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



import logging
import os
from datetime import datetime

def setup_logger():
    """返回一个带有时间戳的日志对象"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在

    # 生成带有时间的日志文件
    log_filename = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    logger = logging.getLogger("BERT_Classifier")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    
    return logger