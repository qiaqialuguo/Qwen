import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_path = '/opt/large-model/deepseek/deepseek-llm-7b-chat'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                             device_map="cuda"
,bnb_4bit_compute_dtype=torch.float16
, load_in_4bit=True
                                             )
model.generation_config = GenerationConfig.from_pretrained(model_path)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

messages = [
    {"role": "user", "content": """15个圆球从上往下排列，其中只有1个是红色的，从上往下数，红色圆球位于第6个，这时，从最下面拿走一个球，此时，请问从下往上数，红色圆球在第几个？"""}
]
start_time = time.time()
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=10000)

response = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)
end_time = time.time()
print('生成耗时：', end_time - start_time, '文字长度：', len(response), '每秒字数：',
              len(response) / (end_time - start_time))