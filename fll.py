import datetime
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
# path = '../Qwen-7B-Chat'
path = '/opt/large-model/gemma/gemma-7b'
# 可选的模型包括: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
# 默认使用自动模式，根据设备自动选择精度
model = AutoModelForCausalLM.from_pretrained(path
, device_map="cuda"
, trust_remote_code=True
,bnb_4bit_compute_dtype=torch.float16
, load_in_4bit=True
            ).eval()
# model = model.to("cuda:0")
# 可指定不同的生成长度、top_p等相关超参
model.generation_config = GenerationConfig.from_pretrained(path, trust_remote_code=True)

prompt = '转换成阿拉伯数字：一三八六六六六七七七七'
prompt = '讲一个100字的故事'
import time
print(time.time())


print('start:'+str(datetime.datetime.now()))
r0, _ = model.chat(query=prompt, tokenizer=tokenizer, history=None)
print('end:'+str(datetime.datetime.now()))
print(r0)
print(time.time())
print('end:'+str(datetime.datetime.now()))