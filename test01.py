import os
model_name = "uer/gpt2-chinese-cluecorpussmall"
model_dir = os.path.join(r"C:\Hugging Face\models", model_name)

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

output = generator(
    "你好，我是一款语言模型",
    max_new_tokens=256,
    truncation=True,
    num_return_sequences=1,
    temperature=0.7,
    top_k=500,
    top_p=0.9,
    clean_up_tokenization_spaces=True
)

print(output)