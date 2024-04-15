import datetime
import time
from contextlib import asynccontextmanager

import GPUtil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from rag import rag_args
from rag.rag_handler import ChatCompletionRequest, parse_messages, ChatCompletionResponse, \
    ChatCompletionResponseChoice, ChatMessage, predict, _gc


# * 3.1.1 处理GC
@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    _gc(args=args,forced=True)


# * 3.1 创建FastAPI
app = FastAPI(lifespan=lifespan)

# * 3.1.2 处理跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# * 3.1.3 接口接收数据
@app.post('/v1/chat/completions', response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    # * 3.1.3.-1.处理参数
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

    stop_words = []
    if request.use_rag:
        stop_words = stop_words or []
        if 'Observation:' not in stop_words:
            stop_words.append('Observation:')
        if 'Observation:\n' not in stop_words:
            stop_words.append('Observation:\n')
    global history_global
    # * 3.1.3.0.处理消息
    query, history, system = parse_messages(request.messages, request.user_id,history_global)

    # * 3.1.3.1如果是流式
    if request.stream:
        if not request.use_rag:
            print("\033[1;42m用户【" + request.user_id + "】开始提问，生成答案中...  \033[0m\033[1;45m" + str(
                datetime.datetime.now()) +
                  "  \033[0m\033[1;44m模式：流式，不使用rag\033[0m")
            start_time = time.time()
            start_mem = GPUtil.getGPUs()[0].memoryUsed

            generate = predict(query,
                               history,
                               request.model,
                               stop_words,
                               gen_kwargs,
                               system=system,model=model,tokenizer=tokenizer,args=args)
            print(generate)
            return EventSourceResponse(generate, media_type='text/event-stream')

            # end_time = time.time()
            # end_mem = GPUtil.getGPUs()[0].memoryUsed
            # print("\033[0;33m问题：【" + query + "】\033[0m\n"
            #                                    "\033[0;37m历史:\n[" + str(
            #     ''.join([str(item) + "\n" for item in history])[:-1]) + "]\033[0m\n"
            #                                                             "\033[0;36m回答：【" + response + "】\033[0m")
            # print('\033[1;44m回答完毕，耗时：', end_time - start_time, '答案长度：', len(response), '每秒字数：',
            #       '时间没变' if end_time == start_time else len(response) / (end_time - start_time), '输入长度:',
            #       len(query), '显存增加:',
            #       (end_mem - start_mem) / 1024, 'G\033[0m')

            # * 3.1.3.1.1. 记录历史
            history.append((query, response))
            history_global[request.user_id] = history
    # * 3.1.3.2如果是非流式
    else:
        # * 3.1.3.2.1 如果不用rag
        if not request.use_rag:
            print("\033[1;42m用户【" + request.user_id + "】开始提问，生成答案中...  \033[0m\033[1;45m" + str(
                datetime.datetime.now()) +
                  "  \033[0m\033[1;44m模式：非流式，不使用rag\033[0m")
            start_time = time.time()
            start_mem = GPUtil.getGPUs()[0].memoryUsed
            response, _ = model.chat(tokenizer, query=query, history=history, stop_words_ids=stop_words,
                                     system=system)
            end_time = time.time()
            end_mem = GPUtil.getGPUs()[0].memoryUsed
            print("\033[0;33m问题：【" + query + "】\033[0m\n"
                                               "\033[0;37m历史:\n[" + str(
                ''.join([str(item) + "\n" for item in history])[:-1]) + "]\033[0m\n"
                                                                        "\033[0;36m回答：【" + response + "】\033[0m")
            print('\033[1;44m回答完毕，耗时：', end_time - start_time, '答案长度：', len(response), '每秒字数：',
                  '时间没变' if end_time == start_time else len(response) / (end_time - start_time), '输入长度:',
                  len(query), '显存增加:',
                  (end_mem - start_mem) / 1024, 'G\033[0m')

            # * 3.1.3.2.1.1 记录历史
            history.append((query, response))
            history_global[request.user_id] = history

            choice_data = ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role='assistant', content=response),
                finish_reason='stop',
            )
            return ChatCompletionResponse(model=request.model,
                                          choices=[choice_data],
                                          object='chat.completion')
        # * 3.1.3.2.2 如果用rag
        else:
            prompt = build_planning_prompt(TOOLS, query)  # 组织prompt
            model.generation_config.do_sample = False  # greedy 禁用采样，贪婪
            response, _ = model.chat(tokenizer, prompt, history=history_tmp,
                                     stop_words_ids=stop_words_ids)
            while "Final Answer:" not in response:  # 出现final Answer时结束
                api_output = use_api(TOOLS, response)  # 抽取入参并执行api
                api_output = str(api_output)  # 部分api工具返回结果非字符串格式需进行转化后输出
                if "no tool founds" == api_output:
                    break
                print("\033[32m" + response + "\033[0m" + "\033[34m" + ' ' + api_output + "\033[0m")
                prompt = prompt + response + ' ' + api_output  # 合并api输出
                response, _ = model.chat(tokenizer, prompt, history=history_tmp, stop_words_ids=stop_words,
                                         system=system)  # 继续生成
            # print("\033[32m" + response + "\033[0m")
            print('<chat>')
            pprint(history_tmp, indent=2)
            print(f'{query}\n<!-- *** -->\n{response}\n</chat>')
    _gc(args=args)
    response = response.split('Final Answer:')[-1]
    history_tmp.append((query, response))
    if len(history_tmp) > 7:
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


if __name__ == '__main__':
    # * 1.获取参数
    args = rag_args.get_args()
    # * 1.1 定义全局history
    history_global = dict()
    # * 2.加载模型
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,  # 信任从远程位置下载的模型代码
        resume_download=True,  # 支持断点续传
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map='cuda',
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
    # * 3.运行web框架
    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)
