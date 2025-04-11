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

def get_data(sample_size=1600):
    base_dir = os.path.join('data', 'aclImdb_v1', 'aclImdb')

    # 训练集和测试集的路径
    train_pos_path = os.path.join(base_dir, 'train', 'pos')
    train_neg_path = os.path.join(base_dir, 'train', 'neg')
    test_pos_path = os.path.join(base_dir, 'test', 'pos')
    test_neg_path = os.path.join(base_dir, 'test', 'neg')
    
    # 确保文件夹存在
    if not os.path.exists(test_pos_path) or not os.path.exists(test_neg_path) or \
       not os.path.exists(train_pos_path) or not os.path.exists(train_neg_path):
        raise FileNotFoundError("One or more directories are missing!")

    # 读取文件
    pos_all, neg_all = [], []
    
    # 读取测试集
    for filename in os.listdir(test_pos_path):
        with open(os.path.join(test_pos_path, filename), encoding='utf8') as f:
            pos_all.append(f.read())
    
    for filename in os.listdir(test_neg_path):
        with open(os.path.join(test_neg_path, filename), encoding='utf8') as f:
            neg_all.append(f.read())

    # 读取训练集
    for filename in os.listdir(train_pos_path):
        with open(os.path.join(train_pos_path, filename), encoding='utf8') as f:
            pos_all.append(f.read())
    
    for filename in os.listdir(train_neg_path):
        with open(os.path.join(train_neg_path, filename), encoding='utf8') as f:
            neg_all.append(f.read())

    # 合并数据
    datasets = np.array(pos_all + neg_all)
    labels = np.array([1] * len(pos_all) + [0] * len(neg_all))
    return datasets[:sample_size], labels[:sample_size]  # 截取指定数量


def shuffle_process():
    sentences, labels = get_data()
    # Shuffle
    shuffle_indexs = np.random.permutation(len(sentences))
    datasets = sentences[shuffle_indexs]
    labels = labels[shuffle_indexs]
    return datasets, labels


def save_process():
    datasets, labels = shuffle_process()
    sentences = []
    
    # 使用正则表达式清理文本
    punc = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]'
    for sen in datasets:
        sen = sen.replace('\n', ' ')
        sen = sen.replace('<br /><br />', ' ')
        sen = re.sub(punc, '', sen)
        sentences.append(sen)
    
    # Save
    df = pd.DataFrame({'labels': labels, 'sentences': sentences})
    df.to_csv("data/datasets.csv", index=False)


class IMDBDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.sentences = self.data["sentences"].tolist()
        self.labels = self.data["labels"].tolist()
        tokenizer = AutoTokenizer.from_pretrained("./model/google-bert/bert-base-uncased")
        self.encodings = tokenizer(
            self.sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        print(self.encodings )

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
    dataset = IMDBDataset("data/datasets.csv")

    # 划分训练集和测试集（80% 训练，20% 测试）
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader   


if __name__ == '__main__':
    save_process()
    train_loader, test_loader = get_dataloaders()
    print(f"训练集批次数: {len(train_loader)}, 测试集批次数: {len(test_loader)}")