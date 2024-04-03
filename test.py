from elasticsearch import Elasticsearch

# 创建 Elasticsearch 客户端
es = Elasticsearch([{'host': '223.203.3.210', 'port': 9201, 'scheme': 'my-application'}]
                   , basic_auth=('elastic', 'enRh$tQ!oV8DB*Uy'))

# 定义模糊查询 DSL
query = {
    "query": {
        "fuzzy": {
            "car_name": {
                "value": "奥迪Q3 2018款 31周年",
                "fuzziness": "auto"
            }
        }
    }
}

# 执行查询
result = es.search(index="car_price", body=query)
print(result)
# 处理查询结果
for hit in result['hits']['hits']:
    print(hit['_score'], hit['_source'])


import requests
response = requests.get('http://223.203.3.210:9201/car_price/_analyzer', auth=('elastic', 'enRh$tQ!oV8DB*Uy'))

# 输出响应
print(response.json())