# pip install langchain google-search-results
import json
import os
from typing import Tuple

import torch
from langchain import SerpAPIWrapper
from langchain_experimental.tools import PythonAstREPLTool
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 为了使用谷歌搜索（SERPAPI）， 您需要自行申请它们的 API KEY，然后填入此处
os.environ['SERPAPI_API_KEY'] = 'a89827b916b3fbc639d06b01eeaf585d83131bf1070ec2fb937cb22639b66339'

search = SerpAPIWrapper()
python = PythonAstREPLTool()


def tool_wrapper_for_qwen(tool):
    def tool_(query):
        query = json.loads(query)["query"]
        return tool.run(query)

    return tool_

def tool_wrapper_for_qwen_price():
    def tool_(query):
        query = json.loads(query)["query"]
        return '价格是21.59万'
    return tool_

def tool_wrapper_for_qwen_configuration():
    def tool_(query):
        query = json.loads(query)["query"]
        return '车辆配置包括LED大灯、全景天窗、电动尾门、真皮座椅、自动空调，价格是21.59万'
    return tool_

# 以下是给千问看的工具描述：
TOOLS = [
    # {
    #     'name_for_human':
    #         'google search',
    #     'name_for_model':
    #         'Search',
    #     'description_for_model':
    #         'useful for when you need to answer questions about current events.',
    #     'parameters': [{
    #         "name": "query",
    #         "type": "string",
    #         "description": "search query of google",
    #         'required': True
    #     }],
    #     'tool_api': tool_wrapper_for_qwen(search)
    # },
    {
        'name_for_human':
            'python',
        'name_for_model':
            'python',
        'description_for_model':
            "A Python shell. Use this to execute python commands include Math,current time,date or day of the week. When using this tool, sometimes output is abbreviated - Make sure it does not look abbreviated before using it in your answer. "
            "Don't add comments to your python code.",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "a valid python command.",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen(python)
    },
    {
        'name_for_human':
            'the_car_price',
        'name_for_model':
            'the_car_price',
        'description_for_model':
            "A database of car price. Use this to search the car price.只有查询车辆价格时才使用这个工具",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "汽车的年款，品牌和车系和是新车还是二手车，默认2024款，新车",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen_price()
    },
    {
        'name_for_human':
            'the_car_price',
        'name_for_model':
            'the_car_price',
        'description_for_model':
            "A database of car configuration. Use this to search the car configuration.询问车辆配置时使用这个工具",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "汽车的年款，品牌和车系，默认2024款",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen_configuration()
    }

]
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action with json formatted
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""


def build_planning_prompt(TOOLS, query):
    #  ensure_ascii=False：非ascii不会被转义
    tool_descs = []
    tool_names = []
    for info in TOOLS:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model=info['name_for_model'],
                name_for_human=info['name_for_human'],
                description_for_model=info['description_for_model'],
                parameters=json.dumps(
                    info['parameters'], ensure_ascii=False),
            )
        )
        tool_names.append(info['name_for_model'])
    tool_descs = '\n\n'.join(tool_descs)
    tool_names = ','.join(tool_names)

    prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=query)
    return prompt


# prompt_1 = build_planning_prompt(TOOLS[0:1], query="加拿大2023年人口统计数字是多少？")
# print(prompt_1)



# stop = ["Observation:", "Observation:\n"]
# react_stop_words_tokens = [TOKENIZER.encode(stop_) for stop_ in stop]
# response_1, _ = MODEL.chat(TOKENIZER, prompt_1, history=None, stop_words_ids=react_stop_words_tokens)
# print(response_1)


def parse_latest_plugin_call(text: str) -> Tuple[str, str]:
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
    if 0 <= i < j < k:
        plugin_name = text[i + len('\nAction:'):j].strip()
        plugin_args = text[j + len('\nAction Input:'):k].strip()
        return plugin_name, plugin_args
    return '', ''


def use_api(tools, response):
    use_toolname, action_input = parse_latest_plugin_call(response)
    if use_toolname == "":
        return "no tool founds"

    used_tool_meta = list(filter(lambda x: x["name_for_model"] == use_toolname, tools))
    if len(used_tool_meta) == 0:
        return "no tool founds"
    print('使用的工具：' + used_tool_meta[0]["name_for_model"])
    api_output = used_tool_meta[0]["tool_api"](action_input)
    return api_output


# api_output = use_api(TOOLS, response_1)
# print(api_output)
#
# prompt_2 = prompt_1 + response_1 + ' ' + api_output
# stop = ["Observation:", "Observation:\n"]
# react_stop_words_tokens = [TOKENIZER.encode(stop_) for stop_ in stop]
# response_2, _ = MODEL.chat(TOKENIZER, prompt_2, history=None, stop_words_ids=react_stop_words_tokens)
# print(prompt_2, response_2)

checkpoint = "/opt/large-model/qwen/qwen1/Qwen-14B-Chat"
TOKENIZER = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
MODEL = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="cuda", trust_remote_code=True,
                                             bnb_4bit_compute_dtype=torch.float16
                                             , load_in_4bit=True).eval()
MODEL.generation_config = GenerationConfig.from_pretrained(checkpoint, trust_remote_code=True)
MODEL.generation_config.do_sample = False  # greedy 禁用采样，贪婪

def main(query, choose_tools):
    prompt = build_planning_prompt(choose_tools, query)  # 组织prompt
    print(prompt)
    stop = ["Observation:", "Observation:\n"]
    react_stop_words_tokens = [TOKENIZER.encode(stop_) for stop_ in stop]
    response, _ = MODEL.chat(TOKENIZER, prompt, history=None, stop_words_ids=react_stop_words_tokens)

    while "Final Answer:" not in response:  # 出现final Answer时结束
        print(response)
        api_output = use_api(choose_tools, response)  # 抽取入参并执行api
        api_output = str(api_output)  # 部分api工具返回结果非字符串格式需进行转化后输出
        if "no tool founds" == api_output:
            break
        print("\033[32m" + response + "\033[0m" + "\033[34m" + ' ' + api_output + "\033[0m")
        prompt = prompt + response + ' ' + api_output  # 合并api输出
        response, _ = MODEL.chat(TOKENIZER, prompt, history=None, stop_words_ids=react_stop_words_tokens)  # 继续生成

    print("\033[32m" + response + "\033[0m")


# 请尽可能控制备选工具数量
# query = "小米汽车su7多少钱" # 所提问题
# choose_tools = TOOLS # 选择备选工具
# print("=" * 10)
# main(query, choose_tools)

# query = "求解方程 2x+5 = -3x + 7" # 所提问题
# choose_tools = TOOLS # 选择备选工具
# print("=" * 10)
# main(query, choose_tools)


# query = "使用python对下面的列表进行排序： [2, 4135, 523, 2, 3]"
query = '计算 1+1'
choose_tools = TOOLS  # 选择备选工具
print("=" * 10)
main(query, choose_tools)
