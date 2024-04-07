import time

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import transformers
import torch

model_id = "/opt/large-model/glm/chatglm3-6b/"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype
,bnb_4bit_compute_dtype=torch.float16
, load_in_4bit=True
,trust_remote_code=True
)

chat = [
    { "role": "user", "content": """15个圆球从上往下排列，其中只有1个是红色的，从上往下数，红色圆球位于第6个，这时，从最下面拿走一个球，此时，请问从下往上数，红色圆球在第几个？""" },
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False,
                                       add_generation_prompt=True,device_map='cuda')
inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
start_time = time.time()
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=1500)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
end_time = time.time()
print('生成耗时：', end_time - start_time, '文字长度：', len(response), '每秒字数：',
              len(response) / (end_time - start_time))