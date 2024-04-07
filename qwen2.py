import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "../../qwen1.5/Qwen1.5-0.5B-Chat",
    device_map="cuda"
,bnb_4bit_compute_dtype=torch.float16
, load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained("../../qwen1.5/Qwen1.5-0.5B-Chat")

prompt = """讲个故事，请在50字内回答"""
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
start_time = time.time()
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
end_time = time.time()
print('生成耗时：', end_time - start_time, '文字长度：', len(response), '每秒字数：',
              len(response) / (end_time - start_time))