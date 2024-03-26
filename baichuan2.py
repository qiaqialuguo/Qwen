import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
model_dir = '/opt/large-model/baichuan/Baichuan2-7B-Chat/'
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto",
                              trust_remote_code=True, torch_dtype=torch.float16
                                          , bnb_4bit_compute_dtype=torch.float16
                                          , load_in_4bit=True
                                          )
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",
                              trust_remote_code=True, torch_dtype=torch.bfloat16
,bnb_4bit_compute_dtype=torch.float16
, load_in_4bit=True
                                        )
model.generation_config = GenerationConfig.from_pretrained(model_dir)
messages = []
messages.append({"role": "user", "content": """讲个故事，请在50字内回答"""})
start_time = time.time()
response = model.chat(tokenizer, messages)
print(response)
end_time = time.time()
print('生成耗时：', end_time - start_time, '文字长度：', len(response), '每秒字数：',
              len(response) / (end_time - start_time))