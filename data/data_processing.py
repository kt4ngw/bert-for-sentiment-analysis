import csv
import os
import re
 
import numpy as np
import torch

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from transformers import AutoTokenizer
# 加载 BERT 预训练的 Tokenizer
from datasets import load_dataset, load_from_disk

def download_and_save_imdb():
    if not os.path.exists("data/imdb"):  # 避免重复下载
        print("Downloading IMDB dataset...")
        dataset = load_dataset("imdb")
        dataset.save_to_disk("data/imdb")
    else:
        print("IMDB dataset already exists.")



class IMDBDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        dataset = load_from_disk("data/imdb")[split]  # 读取训练或测试集
        raw_texts = dataset["text"][:max_samples] if max_samples else dataset["text"]
        self.labels = dataset["label"][:max_samples] if max_samples else dataset["label"]
        self.texts = self.clean_texts(raw_texts)


        tokenizer = AutoTokenizer.from_pretrained("./model/google-bert/bert-base-uncased")
        self.encodings = tokenizer(
            self.texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
    def clean_texts(self, texts):
        """清理文本数据"""
        cleaned_texts = []
        punc = r"[’!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\n。！，]"  # 正则表达式匹配所有标点符号

        for sen in texts:
            sen = sen.replace("<br /><br />", " ")  # 去除 HTML 换行符
            sen = re.sub(punc, "", sen)  # 删除所有标点符号
            sen = sen.strip()  # 去除首尾空格
            cleaned_texts.append(sen)

        return cleaned_texts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
def get_dataloaders(batch_size=16):
    """创建训练和测试 DataLoader"""
    download_and_save_imdb()  # 先下载数据集

    train_dataset = IMDBDataset("train", max_samples=1600)
    test_dataset = IMDBDataset("test", max_samples=1600)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader   


if __name__ == '__main__':
    
    train_loader, test_loader = get_dataloaders()
    print(f"训练集批次数: {len(train_loader)}, 测试集批次数: {len(test_loader)}")