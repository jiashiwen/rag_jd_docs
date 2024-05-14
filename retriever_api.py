from fastapi import FastAPI
from pydantic import BaseModel
from singleton_decorator import singleton
from langchain_community.embeddings import HuggingFaceEmbeddings
import langchain_community.vectorstores.clickhouse as clickhouse
import uvicorn

app = FastAPI()
app.host = "0.0.0.0"

model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(
    model_name="/root/models/moka-ai-m3e-large", model_kwargs=model_kwargs)
settings = clickhouse.ClickhouseSettings(
    table="jd_docs_m3e", username="default", password="Git785230", host="10.0.1.94")
ck_db = clickhouse.Clickhouse(embeddings, config=settings)
ck_retriver = ck_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 1})


class question(BaseModel):
    content: str


@app.get("/")
async def root():
    return {"ok"}


@app.post("/retriver")
async def retriver(question: question):
    global ck_retriver
    return ck_retriver.invoke(question.content)

if __name__ == '__main__':
    uvicorn.run(app='retriever_api:app', host="0.0.0.0",
                port=8000, reload=True)
