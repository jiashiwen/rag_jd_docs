from fastapi import FastAPI
from pydantic import BaseModel
from singleton_decorator import singleton
from langchain_community.embeddings import HuggingFaceEmbeddings
import langchain_community.vectorstores.clickhouse as clickhouse
import uvicorn
import json

app = FastAPI()
app = FastAPI(docs_url=None)
app.host = "0.0.0.0"

model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(
    model_name="/root/models/moka-ai-m3e-large", model_kwargs=model_kwargs)
settings = clickhouse.ClickhouseSettings(
    table="jd_docs_m3e_with_url_splited", username="default", password="Git785230", host="10.0.1.94")
ck_db = clickhouse.Clickhouse(embeddings, config=settings)
ck_retriever = ck_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})


class question(BaseModel):
    content: str


@app.get("/")
async def root():
    return {"ok"}


@app.post("/retriever")
async def retriver(question: question):
    global ck_retriever
    result = ck_retriever.invoke(question.content)
    return result


if __name__ == '__main__':
    uvicorn.run(app='retriever_api:app', host="0.0.0.0",
                port=8000, reload=True)
