# ToDo

- [x] 将获取prompt 和 通过llm 推理拆成两个服务
- [] retriever 是否支持分值
- [ ] langchain 按 token输出







curl --request POST \
  --url http://127.0.0.1:8888/answer \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: insomnia/9.1.1' \
  --data '{"content":"京东云的clickhouse如何配置"}'