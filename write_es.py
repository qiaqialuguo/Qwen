import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers

import pymongo

# 建立MongoDB连接
client = pymongo.MongoClient('mongodb://localhost:27017/')  # 替换成你的MongoDB连接地址

# 选择数据库
db = client['play']  # 替换成你的数据库名称

collection = db['dongchedi_car_list_1']

query = {}  # 查询条件，这里为空表示查询所有文档
projection = {"car_name": 1, "city": 1, "money": 1,'mileage':1}  # 选择要返回的字段，1表示返回，0表示不返回

# 执行查询
cursor = collection.find(query, projection)

# 将查询结果存入DataFrame
df = pd.DataFrame(list(cursor))
pd.set_option('display.max_columns', None)  # 设置显示所有列
print(df)

# 创建Elasticsearch客户端
es = Elasticsearch([{'host': '223.203.3.210', 'port': 9201,'scheme':'my-application'}]
                   ,basic_auth=('elastic', 'enRh$tQ!oV8DB*Uy'))

# 定义映射（schema）
mapping = {
    "mappings": {
        "properties": {
            "car_name": {"type": "keyword"},
            "city": {"type": "keyword"},
            "money": {"type": "keyword"},
            "mileage": {"type": "keyword"}
        }
    }
}

# 创建索引
index_name = "car_price"  # 索引名称
# 清空索引
es.indices.delete(index=index_name)
es.indices.create(index=index_name, body=mapping)

# 将数据转换为批量写入的格式
data = []
for index, row in df.iloc[:,1:].iterrows():
    print(index)
    doc = {
        "_index": "car_price",  # 替换成你的索引名称
        "_id": str(index),  # 使用数据的行号作为文档ID
        "_source": row.to_dict()
    }
    data.append(doc)

# 执行批量写入
helpers.bulk(es, data)