import json

import requests


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