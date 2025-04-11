import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, test_loader, device, logger=None):
    """评估模型在测试集上的表现"""
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

            if logger and i % 10 == 0:  # 每 10 个 batch 记录一次
                logger.info(f"Evaluating batch {i+1}/{len(test_loader)}...")

    acc = accuracy_score(labels_all, preds_all)
    
    if logger:
        logger.info(f"Test Accuracy: {acc:.4f}")
    else:
        print(f"Test Accuracy: {acc:.4f}")

    return acc
            
