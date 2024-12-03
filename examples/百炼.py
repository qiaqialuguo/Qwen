import os
import dashscope

messages = [
    {'role':'system','content':'you are a helpful assistant'},
    {'role': 'user','content': '你是谁？'}
    ]
responses = dashscope.Generation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key='sk-dc5e9d4949394662a62a9cffbc2f63b4',
    model="qwen-plus-latest",
    messages=messages,
    result_format='message',
    stream=True,
    incremental_output=True
    )
for response in responses:
    print(response)