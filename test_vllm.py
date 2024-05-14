import os
os.environ["http_proxy"] = "http://127.0.0.1:1083"
os.environ["https_proxy"] = "http://127.0.0.1:1083"
import transformers
from transformers import AutoModel
import torch
from vllm import LLM

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

modle=AutoModel.from_pretrained(model_id)
llm = LLM(model=modle) 