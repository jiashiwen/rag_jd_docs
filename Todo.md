# ToDo

- [x] 将获取prompt 和 通过llm 推理拆成两个服务
- [ ] retriever 支持分值
- [ ] langchain 按 token输出
- [ ] 验证使用公共大模型api
- [ ] 数据细粒度切分，验证MarkdownHeaderTextSplitter
- [ ] rerank 理解与应用



## Q&A
通过静态PV的方式使用京东云文件服务 检索效果不佳
是否可以通过将标题也作为doc存储的方式进行精细化匹配


curl --request POST \
  --url http://127.0.0.1:8888/answer \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: insomnia/9.1.1' \
  --data '{"content":"京东云的clickhouse如何配置"}'