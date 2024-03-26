# from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("/opt/large-model/mistral/Mistral-7B-Instruct-v0.2"
                                            ,torch_dtype=torch.bfloat16
,bnb_4bit_compute_dtype=torch.float16
, load_in_4bit=True
                                             ,device_map='cuda'
                                             )
tokenizer = AutoTokenizer.from_pretrained("/opt/large-model/mistral/Mistral-7B-Instruct-v0.2")

messages = [
    # {"role": "user", "content": "What is your favourite condiment?"},
    # {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": """转换成阿拉伯数字：一三八六六六六七七七七中文回答"""}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])