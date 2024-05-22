from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import VLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain.prompts import PromptTemplate
import requests
import uvicorn
import json
import logging

app = FastAPI()
app = FastAPI(docs_url=None)
app.host = "0.0.0.0"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
to_console = logging.StreamHandler()
logger.addHandler(to_console)


# load model
# model_name = "/root/models/Llama3-Chinese-8B-Instruct"
model_name = "/root/models/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# llm_llama3 = VLLM(
#     model=model_name,
#     tokenizer=tokenizer,
#     task="text-generation",
#     temperature=0.2,
#     do_sample=True,
#     repetition_penalty=1.1,
#     return_full_text=False,
#     max_new_tokens=900,
# )

llm_llama3 = AutoModelForCausalLM.from_pretrained(model_name, temperature=0.2)
# inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
streamer = TextStreamer(tokenizer)

# prompt
prompt_template = """
你是一个云技术专家
使用以下检索到的Context回答问题。
如果不知道答案，就说不知道。
用中文回答问题。
Question: {question}
Context: {context}
Answer: 
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
    # partial_variables={"format_instructions": output_parser.get_format_instructions()},
)


def get_context_list(q: str):
    url = "http://10.0.0.7:8000/retriever"
    payload = {"content": q}
    res = requests.post(url, json=payload)
    return res.text


class question(BaseModel):
    content: str


@app.get("/")
async def root():
    return {"ok"}


@app.post("/answer")
async def answer(q: question):
    logger.info("invoke!!!")
    global prompt
    global llm_llama3
    global streamer
    global tokenizer
    context_list_str = get_context_list(q.content)
    context_list = json.loads(context_list_str)
    context = ""
    source_list = []

    for context_json in context_list:
        context = context+context_json["page_content"]
        source_list.append(context_json["metadata"]["source"])
    p = prompt.format(context=context, question=q.content)
    # answer = llm_llama3(p)
    inputs = tokenizer([p], return_tensors="pt")
    _ = llm_llama3.generate(**inputs, streamer=streamer,
                            max_new_tokens=200)
    result = {
        "answer": answer,
        "sources": source_list
    }
    return result


if __name__ == '__main__':
    uvicorn.run(app='retriever_api:app', host="0.0.0.0",
                port=8888, reload=True)
