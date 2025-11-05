from transformers import AutoModel, AutoTokenizer
import os
model_name = "bert-base-chinese"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_dir = os.path.join(r"C:\Hugging Face\models", model_name)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
