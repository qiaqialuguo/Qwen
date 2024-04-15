# Requirement:
#   pip install "openai<1.0"
# Usage:
#   python openai_api.py
# Visit http://localhost:8000/docs for documents.

import base64
import copy
import json
import time
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from pprint import pprint
from typing import Dict, List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import json
import os
from typing import Tuple
import requests

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

def tool_wrapper_for_qwen_configuration():
    def tool_(query):
        query = json.loads(query)["query"]
        response = requests.get(f'http://192.168.110.138:9169/customer-service/bava/getCarConfig?params={query}')
        # 处理响应
        if response.status_code == 200:
            # 请求成功
            data = response.json()  # 获取响应数据，如果是 JSON 格式
            return str(data)
        else:
            # 请求失败
            return '查询失败，请检查'
        # return '车辆配置包括LED大灯、全景天窗、电动尾门、真皮座椅、自动空调，价格是21.59万'
    return tool_

def tool_wrapper_for_qwen_price():
    def tool_(query):
        query = json.loads(query)["query"]
        response = requests.get(f'http://192.168.110.138:9169/customer-service/bava/getCarPrice?params={query}')
        # 处理响应
        if response.status_code == 200:
            # 请求成功
            data = response.json()  # 获取响应数据，如果是 JSON 格式
            return str(data)
        else:
            # 请求失败
            return '查询失败，请检查'
        # return '价格是21.59万'
    return tool_

def tool_wrapper_for_qwen_appointment():
    def tool_(query):
        query = json.loads(query)["query"]
        # if (query == '保养' or query == '维修' or query == '修车'):
        #     query = query_user
        #     response = requests.get(f'http://192.168.110.138:9169/customer-service/bava/appointment?params={query}')
        # else:
        query_user.append(query)
        query = query_user
        print(query)
        response = requests.get(f'http://192.168.110.138:9169/customer-service/bava/appointment?params={query}')
        # 处理响应
        if response.status_code == 200:
        #     请求成功
        #     data = response.json()  # 获取响应数据，如果是 JSON 格式
            return response.text
        else:
            # 请求失败
            return '抱歉，记录失败'
    return tool_

def tool_wrapper_for_qwen_name():
    def tool_(query):
        query = json.loads(query)["query"]
        return '你是宝马智能机器人BAVA，你是ubiai开发的'
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
    # {
    #     'name_for_human':
    #         'python',
    #     'name_for_model':
    #         'python',
    #     'description_for_model':
    #         "A Python shell. Use this to execute python commands include Math,current time,date or day of the week. When using this tool, sometimes output is abbreviated - Make sure it does not look abbreviated before using it in your answer. "
    #         "Don't add comments to your python code.",
    #     'parameters': [{
    #         "name": "query",
    #         "type": "string",
    #         "description": "a valid python command.",
    #         'required': True
    #     }],
    #     'tool_api': tool_wrapper_for_qwen(python)
    # },
    {
        'name_for_human':
            'the_car_price',
        'name_for_model':
            'the_car_price',
        'description_for_model':
            "A database of car's price. 使用这个工具查询车辆的价格(price,how much，多少钱)，只在明确说到车辆的价格时使用这个工具,只说车型时不用这个工具，用查询配置的工具,价格很多时整理成一个价格范围",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "汽车的年款,品牌,车系，年款默认2024款",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen_price()
    },
    {
        'name_for_human':
            'the_car_configuration',
        'name_for_model':
            'the_car_configuration',
        'description_for_model':
            "A database of car's configuration. 使用这个工具查询车辆的配置(参数，车辆信息)，在查询车辆的配置时使用这个工具，只提供车系的话用这个工具,配置信息比较多时，整理成一句话。",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "汽车的年款,品牌,车系，年款默认2024款",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen_configuration()
    },
    {
        'name_for_human':
            'the_car_appointment',
        'name_for_model':
            'the_car_appointment',
        'description_for_model':
            "记录user的预约信息. 使用这个工具记录用户的预约信息，至少要有time和预约的项目，时间要具体到哪一天和几点钟，小于12点的话要继续问上午还是下午还是晚上，项目包括保养（不需要问具体的保养项目）和维修（修车和换零件都属于维修），"
            "也就是预约干什么，维修或保养二选一即可，不需要问具体维修什么或保养什么，如果用户没说time或项目就让用户在一句话中描述时间和项目,如果问去哪里方便吗也是预约，If the user's response is not complete in terms of which day or what time, ask them for the time，收集全时间和项目后，调用这个工具。",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "query为json的key,value是time（What day and what time）和项目（项目包含保养或维修），用空格分割",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen_appointment()
    },
    {
        'name_for_human':
            'name',
        'name_for_model':
            'name',
        'description_for_model': "返回AI助手的名字. 当用户问到你是谁或者你是谁开发的，或者你是不是谁时，使用这个工具，只问名字时不用说是谁开发的，根据用户是问的中英文来选择是中文还是英文回答，你是宝马智能机器人BAVA",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "name",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen_name()
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


class BasicAuthMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, username: str, password: str):
        super().__init__(app)
        self.required_credentials = base64.b64encode(
            f'{username}:{password}'.encode()).decode()

    async def dispatch(self, request: Request, call_next):
        authorization: str = request.headers.get('Authorization')
        if authorization:
            try:
                schema, credentials = authorization.split()
                if credentials == self.required_credentials:
                    return await call_next(request)
            except ValueError:
                pass

        headers = {'WWW-Authenticate': 'Basic'}
        return Response(status_code=401, headers=headers)


def _gc(forced: bool = False):
    global args
    if args.disable_gc and not forced:
        return

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    _gc(forced=True)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class ModelCard(BaseModel):
    id: str
    object: str = 'model'
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = 'owner'
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = 'list'
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system', 'function']
    content: Optional[str]
    function_call: Optional[Dict] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal['user', 'assistant', 'system']] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Union[ChatMessage]
    finish_reason: Literal['stop', 'length', 'function_call']


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal['stop', 'length']]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal['chat.completion', 'chat.completion.chunk']
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get('/v1/models', response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id='gpt-3.5-turbo')
    return ModelList(data=[model_card])


# To work around that unpleasant leading-\n tokenization issue!
def add_extra_stop_words(stop_words):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip('\n')
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    return stop_words


def trim_stop_words(response, stop_words):
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response


TOOL_DESC = (
    '{name_for_model}: Call this tool to interact with the {name_for_human} API.'
    ' What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}'
)

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

_TEXT_COMPLETION_CMD = object()


def parse_messages(messages, functions):
    if all(m.role != 'user' for m in messages):
        raise HTTPException(
            status_code=400,
            detail='Invalid request: Expecting at least one user message.',
        )

    messages = copy.deepcopy(messages)
    if messages[0].role == 'system':
        system = messages.pop(0).content.lstrip('\n').rstrip()
    else:
        system = 'You are a helpful assistant.'

    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:
            name = func_info.get('name', '')
            name_m = func_info.get('name_for_model', name)
            name_h = func_info.get('name_for_human', name)
            desc = func_info.get('description', '')
            desc_m = func_info.get('description_for_model', desc)
            tool = TOOL_DESC.format(
                name_for_model=name_m,
                name_for_human=name_h,
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info['parameters'],
                                      ensure_ascii=False),
            )
            tools_text.append(tool)
            tools_name_text.append(name_m)
        tools_text = '\n\n'.join(tools_text)
        tools_name_text = ', '.join(tools_name_text)
        instruction = (REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        ).lstrip('\n').rstrip())
    else:
        instruction = ''

    messages_with_fncall = messages
    messages = []
    for m_idx, m in enumerate(messages_with_fncall):
        role, content, func_call = m.role, m.content, m.function_call
        content = content or ''
        content = content.lstrip('\n').rstrip()
        if role == 'function':
            if (len(messages) == 0) or (messages[-1].role != 'assistant'):
                raise HTTPException(
                    status_code=400,
                    detail=
                    'Invalid request: Expecting role assistant before role function.',
                )
            messages[-1].content += f'\nObservation: {content}'
            if m_idx == len(messages_with_fncall) - 1:
                # add a prefix for text completion
                messages[-1].content += '\nThought:'
        elif role == 'assistant':
            if len(messages) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=
                    'Invalid request: Expecting role user before role assistant.',
                )
            if func_call is None:
                if functions:
                    content = f'Thought: I now know the final answer.\nFinal Answer: {content}'
            else:
                f_name, f_args = func_call['name'], func_call['arguments']
                if not content.startswith('Thought:'):
                    content = f'Thought: {content}'
                content = f'{content}\nAction: {f_name}\nAction Input: {f_args}'
            if messages[-1].role == 'user':
                messages.append(
                    ChatMessage(role='assistant',
                                content=content.lstrip('\n').rstrip()))
            else:
                messages[-1].content += '\n' + content
        elif role == 'user':
            messages.append(
                ChatMessage(role='user',
                            content=content.lstrip('\n').rstrip()))
        else:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid request: Incorrect role {role}.')

    query = _TEXT_COMPLETION_CMD
    if messages[-1].role == 'user':
        query = messages[-1].content
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        raise HTTPException(status_code=400, detail='Invalid request')

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i].role == 'user' and messages[i + 1].role == 'assistant':
            usr_msg = messages[i].content.lstrip('\n').rstrip()
            bot_msg = messages[i + 1].content.lstrip('\n').rstrip()
            if instruction and (i == len(messages) - 2):
                usr_msg = f'{instruction}\n\nQuestion: {usr_msg}'
                instruction = ''
            history.append([usr_msg, bot_msg])
        else:
            raise HTTPException(
                status_code=400,
                detail=
                'Invalid request: Expecting exactly one user (or function) role before every assistant role.',
            )
    if instruction:
        assert query is not _TEXT_COMPLETION_CMD
        query = f'{instruction}\n\nQuestion: {query}'
    return query, history, system


def parse_response(response):
    func_name, func_args = '', ''
    i = response.find('\nAction:')
    j = response.find('\nAction Input:')
    k = response.find('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + '\nObservation:'  # Add it back.
        k = response.find('\nObservation:')
        func_name = response[i + len('\nAction:'):j].strip()
        func_args = response[j + len('\nAction Input:'):k].strip()

    if func_name:
        response = response[:i]
        t = response.find('Thought: ')
        if t >= 0:
            response = response[t + len('Thought: '):]
        response = response.strip()
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(
                role='assistant',
                content=response,
                function_call={
                    'name': func_name,
                    'arguments': func_args
                },
            ),
            finish_reason='function_call',
        )
        return choice_data

    z = response.rfind('\nFinal Answer: ')
    if z >= 0:
        response = response[z + len('\nFinal Answer: '):]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role='assistant', content=response),
        finish_reason='stop',
    )
    return choice_data


# completion mode, not chat mode
def text_complete_last_message(history, stop_words_ids, gen_kwargs, system):
    im_start = '<|im_start|>'
    im_end = '<|im_end|>'
    prompt = f'{im_start}system\n{system}{im_end}'
    for i, (query, response) in enumerate(history):
        query = query.lstrip('\n').rstrip()
        response = response.lstrip('\n').rstrip()
        prompt += f'\n{im_start}user\n{query}{im_end}'
        prompt += f'\n{im_start}assistant\n{response}{im_end}'
    prompt = prompt[:-len(im_end)]

    _stop_words_ids = [tokenizer.encode(im_end)]
    if stop_words_ids:
        for s in stop_words_ids:
            _stop_words_ids.append(s)
    stop_words_ids = _stop_words_ids

    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    output = model.generate(input_ids,
                            stop_words_ids=stop_words_ids,
                            **gen_kwargs).tolist()[0]
    output = tokenizer.decode(output, errors='ignore')
    assert output.startswith(prompt)
    output = output[len(prompt):]
    output = trim_stop_words(output, ['<|endoftext|>', im_end])
    print(f'<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>')
    return output

history_tmp = []
query_user = []
@app.post('/v1/chat/completions', response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    gen_kwargs = {}
    if request.top_k is not None:
        gen_kwargs['top_k'] = request.top_k
    if request.temperature is not None:
        if request.temperature < 0.01:
            gen_kwargs['top_k'] = 1  # greedy decoding
        else:
            # Not recommended. Please tune top_p instead.
            gen_kwargs['temperature'] = request.temperature
    if request.top_p is not None:
        gen_kwargs['top_p'] = request.top_p

    stop_words = add_extra_stop_words(request.stop)
    if request.functions:
        stop_words = stop_words or []
        if 'Observation:' not in stop_words:
            stop_words.append('Observation:')

    query, history, system = parse_messages(request.messages,
                                            request.functions)

    if request.stream:
        if request.functions:
            raise HTTPException(
                status_code=400,
                detail=
                'Invalid request: Function calling is not yet implemented for stream mode.',
            )
        print('问题2：'+query)
        generate = predict(query,
                           history,
                           request.model,
                           stop_words,
                           gen_kwargs,
                           system=system)
        return EventSourceResponse(generate, media_type='text/event-stream')


    if stop_words:
        stop_words += ["Observation:", "Observation:\n"]  #  添加停止词
    else:
        stop_words = ["Observation:", "Observation:\n"]
    stop_words_ids = [tokenizer.encode(s)
                      for s in stop_words] if stop_words else None
    if query is _TEXT_COMPLETION_CMD:
        response = text_complete_last_message(history,
                                              stop_words_ids=stop_words_ids,
                                              gen_kwargs=gen_kwargs,
                                              system=system)
    else:
        query_user.append(query)
        if len(query_user) > 3:
            del query_user[:len(query_user) - 3]
        prompt = build_planning_prompt(TOOLS, query)  # 组织prompt
        model.generation_config.do_sample = False  # greedy 禁用采样，贪婪
        response, _ = model.chat(tokenizer, prompt, history=history_tmp,
                                 stop_words_ids=stop_words_ids)
        print('----------------')
        print(response)
        print('----------------')
        while "Final Answer:" not in response:  # 出现final Answer时结束
            api_output = use_api(TOOLS, response)  # 抽取入参并执行api
            api_output = str(api_output)  # 部分api工具返回结果非字符串格式需进行转化后输出
            if "no tool founds" == api_output:
                break
            print("\033[32m" + response + "\033[0m" + "\033[34m" + ' ' + api_output + "\033[0m")
            prompt = prompt + response + ' ' + api_output  # 合并api输出
            response, _ = model.chat(tokenizer, prompt, history=history_tmp, stop_words_ids=stop_words_ids,system=system)  # 继续生成
            print('======================')
            print(response)
            print('======================')

        # print("\033[32m" + response + "\033[0m")
        print('<chat>')
        pprint(history_tmp, indent=2)
        print(f'{query}\n<!-- *** -->\n{response}\n</chat>')
    _gc()
    response = response.split('Final Answer:')[-1]
    history_tmp.append((query,response))
    if len(history_tmp)>7:
        del history_tmp[:len(history_tmp) - 7]

    response = trim_stop_words(response, stop_words)
    if request.functions:
        choice_data = parse_response(response)
    else:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role='assistant', content=response),
            finish_reason='stop',
        )
    return ChatCompletionResponse(model=request.model,
                                  choices=[choice_data],
                                  object='chat.completion')


def _dump_json(data: BaseModel, *args, **kwargs) -> str:
    try:
        return data.model_dump_json(*args, **kwargs)
    except AttributeError:  # pydantic<2.0.0
        return data.json(*args, **kwargs)  # noqa


async def predict(
    query: str,
    history: List[List[str]],
    model_id: str,
    stop_words: List[str],
    gen_kwargs: Dict,
    system: str,
):
    global model, tokenizer
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role='assistant'), finish_reason=None)
    chunk = ChatCompletionResponse(model=model_id,
                                   choices=[choice_data],
                                   object='chat.completion.chunk')
    yield '{}'.format(_dump_json(chunk, exclude_unset=True))

    current_length = 0
    stop_words_ids = [tokenizer.encode(s)
                      for s in stop_words] if stop_words else None

    delay_token_num = max([len(x) for x in stop_words]) if stop_words_ids else 0
    response_generator = model.chat_stream(tokenizer,
                                           query,
                                           history=history,
                                           stop_words_ids=stop_words_ids,
                                           system=system,
                                           **gen_kwargs)
    for _new_response in response_generator:
        if len(_new_response) <= delay_token_num:
            continue
        new_response = _new_response[:-delay_token_num] if delay_token_num else _new_response

        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_text), finish_reason=None)
        chunk = ChatCompletionResponse(model=model_id,
                                       choices=[choice_data],
                                       object='chat.completion.chunk')
        yield '{}'.format(_dump_json(chunk, exclude_unset=True))
    
    if current_length != len(_new_response):
        # Determine whether to print the delay tokens
        delayed_text = _new_response[current_length:]
        new_text = trim_stop_words(delayed_text, stop_words)
        if len(new_text) > 0:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0, delta=DeltaMessage(content=new_text), finish_reason=None)
            chunk = ChatCompletionResponse(model=model_id,
                                        choices=[choice_data],
                                        object='chat.completion.chunk')
            yield '{}'.format(_dump_json(chunk, exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(index=0,
                                                     delta=DeltaMessage(),
                                                     finish_reason='stop')
    chunk = ChatCompletionResponse(model=model_id,
                                   choices=[choice_data],
                                   object='chat.completion.chunk')
    yield '{}'.format(_dump_json(chunk, exclude_unset=True))
    yield '[DONE]'

    _gc()


def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-c',
        '--checkpoint-path',
        type=str,
        default='Qwen/Qwen-7B-Chat',
        help='Checkpoint name or path, default to %(default)r',
    )
    parser.add_argument('--api-auth', help='API authentication credentials')
    parser.add_argument('--cpu-only',
                        action='store_true',
                        help='Run demo with CPU only')
    parser.add_argument('--server-port',
                        type=int,
                        default=8000,
                        help='Demo server port.')
    parser.add_argument(
        '--server-name',
        type=str,
        default='127.0.0.1',
        help=
        'Demo server name. Default: 127.0.0.1, which is only visible from the local computer.'
        ' If you want other computers to access your server, use 0.0.0.0 instead.',
    )
    parser.add_argument(
        '--disable-gc',
        action='store_true',
        help='Disable GC after each response generated.',
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        resume_download=True,
    )

    if args.api_auth:
        app.add_middleware(BasicAuthMiddleware,
                           username=args.api_auth.split(':')[0],
                           password=args.api_auth.split(':')[1])

    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'auto'

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True
        , bnb_4bit_compute_dtype=torch.float16
        , load_in_4bit=True
    ).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        resume_download=True
    )

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)
