
import os
import torch
from torch.optim import AdamW
import torch.nn as nn
from modelscope.hub.snapshot_download import snapshot_download


from data.data_processing import get_dataloaders
from train import train_model
from evaluate import evaluate_model
from utils import get_device, setup_logger
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification



def check_and_download_model(model_id, logger, model_dir="./model/"):
    # 拼接模型的本地路径
    model_path = os.path.join(model_dir, model_id)
    print(model_path)
    # 检查模型是否已存在
    if os.path.exists(model_path):
        logger.info(f"模型路径已存在: {model_path}")
    else:
        logger.info(f"模型路径不存在，开始下载模型...")
        # 使用 ModelScope 下载模型
        model_path = snapshot_download(model_id, cache_dir=model_dir, revision='master')
        logger.info(f"模型已下载到: {model_path}")
    return model_path

def main():
    # **自动下载模型**
    logger = setup_logger()  
    model_id = "google-bert/bert-base-uncased"
    check_and_download_model(model_id, logger)
    device = get_device()
    print(f"Using device: {device}")
    train_loader, test_loader = get_dataloaders(batch_size=32)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    logger.info("开始训练...")
    train_model(model, train_loader, optimizer, criterion, device, epochs=3, logger=logger)
    logger.info("开始评估...")
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()