import datetime
import time

query = "nihao"
history = [('1','2'),(2,4)]
response = 'niyehao'

print("\033[32m" + "开始提问" + "\033[0m")
print("\033[1;42m用户"+ "56" +"开始提问，生成答案中...  \033[0m\033[1;45m" + str(datetime.datetime.now()) +
                  "  \033[0m\033[1;43m模式：非流式，不使用rag\033[0m")

print("\033[0;33m问题：【"+query+"】\033[0m\n"
                  "\033[0;37m历史:\n["+str(''.join([str(item) + "\n" for item in history])[:-1])+"]\033[0m\n"
                  "\033[0;36m回答：【"+response+"】\033[0m")
start_time = time.time()
end_time = time.time()

print('\033[1;44m回答完毕，耗时：', end_time - start_time, '答案长度：', len(response), '每秒字数：',
                  '时间没变' if end_time == start_time else len(response) / (end_time - start_time), '输入长度:', len(query), '显存增加:',
                  (1)/1024, 'G\033[0m')

import GPUtil

# 获取当前可用的 GPU
gpu = GPUtil.getGPUs()[0]

print("GPU显存总量:", gpu.memoryTotal)  # 显存总量
print("GPU显存使用量:", GPUtil.getGPUs()[0].memoryUsed)  # 显存使用量

