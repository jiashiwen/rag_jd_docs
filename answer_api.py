from langchain_community.llms import VLLM
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import VLLM
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate
import requests
import uvicorn

app = FastAPI()
app.host = "0.0.0.0"

# load model
# model_name = "/root/models/Llama3-Chinese-8B-Instruct"
model_name = "/root/models/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_llama3 = VLLM(
    model=model_name,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=900,
)

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


def get_context(q: str):
    url = "http://10.0.0.7:8000/retriver"
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
    global prompt
    global llm_llama3
    context_content = get_context(q.content)
    p = prompt.format(context=context_content, question=q.content)
    result = llm_llama3(p)
    return result.replace("\n", "")


if __name__ == '__main__':
    uvicorn.run(app='retriever_api:app', host="0.0.0.0",
                port=8888, reload=True)
