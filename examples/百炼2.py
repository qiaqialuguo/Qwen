import random
from http import HTTPStatus

from dashscope import Generation

responses = Generation.call(
    "qwen-plus-latest",
    messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': '你是 小优，一个由 优必爱 训练的大型语言模型。知识截止日期：2024-12。当前时间：2024-12-02 18:32:12，今天是星期一。Answer the following questions as best you can. You have access to the following tools:\n\nthe_car_appointment: Call this tool to interact with the the_car_appointment API. What is the the_car_appointment API useful for? 用户预约车辆服务时使用这个工具，工具会在4s店进行真实预约,不要把json格式给用户,不用举格式的例子，你的返回只能有一个FeedbackToUsers，预约信息包括预约时间（appointment_time），车辆维护类型（vehicle_maintenance_type），4s店名称（automobile_sales_service_shop_name），4s店地址（automobile_sales_service_shop_address），用户想要预约时调用这个工具。其中预约时间是必填. Parameters: [{"name": "appointment_time", "type": "string", "description": "预约时间", "required": true}, {"name": "vehicle_maintenance_type", "type": "string", "description": "保养或维修二选一", "required": false}, {"name": "automobile_sales_service_shop_name", "type": "string", "description": "4s店名称", "required": false}, {"name": "automobile_sales_service_shop_address", "type": "string", "description": "4s店地址", "required": false}] Format the arguments as a JSON object.\n\nUse the following format:\n\nUser: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [the_car_appointment]\nAction Input: the input to the action with json formatted\nMonitoring: the result of the action\n\nThought: 我需要将Monitoring的内容返回给用户\nFeedbackToUsers: 返回给用户Monitoring的内容，只返回一次,一定要有这个字段\n\nBegin!\n\nUser: 我要预约\nThought:我将调用the_car_appointment工具来尝试预约\nAction: the_car_appointment\nAction Input:{}\nMonitoring:用户正在预约，需要继续询问用户预约时间\n'}]
,
    seed=random.randint(1, 10000),  # set the random seed, optional, default to 1234 if not set
    result_format='message',  # set the result to be "message"  format.
    stream=True,
    api_key='sk-dc5e9d4949394662a62a9cffbc2f63b4',
    output_in_full=True,  # get streaming output incrementally
    stop=['User:', 'Action:', 'Action Input:', 'Think:', 'Thought:'],
)
current_length = 0
for response in responses:
    if response.status_code == HTTPStatus.OK:
        content = response.output.choices[0]['message']['content']
        if len(content) == current_length:
            continue
        new_text = content[current_length:]
        current_length = len(content)
        # print(new_text, end='', flush=True)
        print(new_text)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
