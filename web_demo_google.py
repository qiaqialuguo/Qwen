# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""
import os
import time
from argparse import ArgumentParser

import gradio as gr
import mdtex2html

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


DEFAULT_CKPT_PATH = ''

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


# 以下是给千问看的工具描述：
TOOLS = [
    {
        'name_for_human':
            'google search',
        'name_for_model':
            'Search',
        'description_for_model':
            'useful for when you need to answer questions about current events.',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "search query of google",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen(search)
    },
    {
        'name_for_human':
            'python',
        'name_for_model':
            'python',
        'description_for_model':
            "A Python shell. Use this to execute python commands. When using this tool, sometimes output is abbreviated - Make sure it does not look abbreviated before using it in your answer. "
            "Don't add comments to your python code.",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "a valid python command.",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen(python)
    }

]
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""


def build_planning_prompt(TOOLS, query):
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

    api_output = used_tool_meta[0]["tool_api"](action_input)
    return api_output

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True
    , bnb_4bit_compute_dtype = torch.float16
    , load_in_4bit = True
    ).eval()

    config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer, config


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _launch_demo(args, model, tokenizer, config):
    def predict(_query, _chatbot, _task_history):
        model.generation_config.do_sample = False  # greedy
        query = _query
        print(TOOLS)
        full_response = ""
        prompt = build_planning_prompt(TOOLS, query)  # 组织prompt
        stop = ["Observation:", "Observation:\n"]
        react_stop_words_tokens = [tokenizer.encode(stop_) for stop_ in stop]
        response = """
        Thought: 我应该使用搜索工具来查找相关信息。
        Action: Search
        Action Input: {"query": " """+ query +""" "}
        Observation:
        """
        print(response)
        if True:
            api_output = use_api(TOOLS, response)  # 抽取入参并执行api
            api_output = str(api_output)  # 部分api工具返回结果非字符串格式需进行转化后输出

            print("\033[32m" + response + "\033[0m" + "\033[34m" + ' ' + api_output + "\033[0m")
            prompt = prompt + response + ' ' + api_output  # 合并api输出

            print(f"User: {_parse_text(_query)}")
            _chatbot.append((_parse_text(_query), ""))

            start_time = time.time()
            for response in model.chat_stream(tokenizer, prompt, history=_task_history, generation_config=config, stop_words_ids=react_stop_words_tokens):
                _chatbot[-1] = (_parse_text(_query), _parse_text(response))

                yield _chatbot
                full_response = _parse_text(response)

            print(f"History: {_task_history}")
            _task_history.append((_query, full_response))
            print(f"Qwen-Chat: {_parse_text(full_response)}")
            end_time = time.time()
            print('生成耗时：', end_time - start_time, '文字长度：', len(_parse_text(full_response)), '每秒字数：',
                  len(_parse_text(full_response)) / (end_time - start_time))


    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks() as demo:
        gr.Markdown("""\
<p align="center"><img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/logo_qwen.jpg" style="height: 80px"/><p>""")
        gr.Markdown("""<center><font size=8>Qwen-Chat Bot</center>""")
        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Qwen-Chat, developed by Alibaba Cloud. \
(本WebUI基于Qwen-Chat打造，实现聊天机器人功能。)</center>""")
        gr.Markdown("""\
<center><font size=4>
Qwen-7B <a href="https://modelscope.cn/models/qwen/Qwen-7B/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen-7B">🤗</a>&nbsp ｜ 
Qwen-7B-Chat <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen-7B-Chat">🤗</a>&nbsp ｜ 
Qwen-14B <a href="https://modelscope.cn/models/qwen/Qwen-14B/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen-14B">🤗</a>&nbsp ｜ 
Qwen-14B-Chat <a href="https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen-14B-Chat">🤗</a>&nbsp ｜ 
&nbsp<a href="https://github.com/QwenLM/Qwen">Github</a></center>""")

        chatbot = gr.Chatbot(label='Qwen-Chat', elem_classes="control-height")
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(predict, [query, chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Qwen. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer, config = _load_model_tokenizer(args)

    _launch_demo(args, model, tokenizer, config)


if __name__ == '__main__':
    main()

