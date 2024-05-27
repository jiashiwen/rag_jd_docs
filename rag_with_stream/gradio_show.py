import json
import gradio as gr
from threading import Thread
import requests
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = "/root/models/Qwen1.5-1.8B-Chat"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


prompt_template = """
你是一个京东云的云计算专家
使用以下检索到的Context，使用中文回答问题。
Context: {context}
如果不知道答案，就说不知道。
"""
prompt = PromptTemplate(
    input_variables=["context"],
    template=prompt_template,
)


def get_context_list(q: str):
    url = "http://10.0.0.7:8000/retriever"
    payload = {"content": q}
    res = requests.post(url, json=payload)
    return res.text


def question_answer(query):
    global tokenizer
    global model
    context_list_str = get_context_list(query)
    context_list = json.loads(context_list_str)
    context = ""
    source_list = []

    for context_json in context_list:
        context = context+context_json["page_content"]
        source_list.append(context_json["metadata"]["source"])
    p = prompt.format(context=context)

    message = [{"role": "system", "content": p},
               {"role": "user", "content": query}]
    conversion = tokenizer.apply_chat_template(
        message, add_generation_prompt=True, tokenize=False)
    encoding = tokenizer(conversion, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer)
    generation_kwargs = dict(encoding, streamer=streamer,
                             max_new_tokens=700, do_sample=True, temperature=0.2)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generate_text = '参考文档：'+"\n" + \
        ''.join('- '+str(e)+"\n" for e in source_list)+"\n"
    for new_text in streamer:
        output = new_text.replace(conversion, '')
        if output:
            generate_text += output
            yield generate_text


demo = gr.Interface(
    fn=question_answer,
    inputs=gr.Textbox(
        lines=3, placeholder="your question...", label="Question"),
    # outputs="text",
    outputs=[gr.Markdown(label="answer")]
)

demo.launch(server_name="0.0.0.0", server_port=8888, share=True)
