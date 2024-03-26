import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '/opt/large-model/yi/Yi-6B-Chat/'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype='auto'
,bnb_4bit_compute_dtype=torch.float16
, load_in_4bit=True
).eval()

# Prompt content: "hi"
messages = [
    {"role": "user", "content": """讲个故事，请在50字内回答"""}
]
start_time = time.time()
input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)
end_time = time.time()
print('生成耗时：', end_time - start_time, '文字长度：', len(response), '每秒字数：',
              len(response) / (end_time - start_time))