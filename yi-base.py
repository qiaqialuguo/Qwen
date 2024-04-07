import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/opt/large-model/yi/Yi-9B"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto",
                                             device_map='cuda'
                                             , bnb_4bit_compute_dtype=torch.float16
                                             , load_in_4bit=True
                                             )
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

input_text = """15个圆球从上往下排列，其中只有1个是红色的，从上往下数，红色圆球位于第6个，这时，从最下面拿走一个球，此时，请问从下往上数，红色圆球在第几个？"""
start_time = time.time()
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=1000)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
end_time = time.time()
print('生成耗时：', end_time - start_time, '文字长度：', len(response), '每秒字数：',
              len(response) / (end_time - start_time))