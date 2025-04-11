# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("fill-mask", model="google-bert/bert-base-uncased")



# # Load model directly
# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")



from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('google-bert/bert-base-uncased', cache_dir='./model/', revision='master')