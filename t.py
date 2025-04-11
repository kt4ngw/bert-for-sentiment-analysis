# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("fill-mask", model="google-bert/bert-base-uncased")



# # Load model directly
# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")


from datasets import load_dataset

def download_and_save_imdb():
    # 加载 IMDB 数据集
    dataset = load_dataset("imdb")
    
    # 保存到本地目录（例如 'data/imdb'）
    dataset.save_to_disk("data/imdb")
# 访问训练集
    train_dataset = dataset["train"]
    print(train_dataset["text"])
    # 访问测试集
    test_dataset = dataset["test"]

    print("训练集大小:", len(train_dataset))
    print("测试集大小:", len(test_dataset))
if __name__ == '__main__':
    download_and_save_imdb()
    print("数据集已保存到 'data/imdb' 目录。")