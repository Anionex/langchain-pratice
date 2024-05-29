from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
result = remote_chain.invoke({"language": "english", "text": "开会时集中注意力并没有那么重要，更重要的是要避免出现任何干扰。"})
print(result)