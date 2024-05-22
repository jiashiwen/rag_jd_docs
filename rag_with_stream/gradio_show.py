# import json
# import gradio as gr
# import requests


# def greet(name, intensity):
#     return "Hello, " + name + "!" * int(intensity)


# def answer(question):
#     url = "http://114.67.87.223:8888/answer"
#     payload = {"content": question}
#     res = requests.post(url, json=payload)
#     res_json = json.loads(res.text)
#     return [res_json["answer"], res_json["sources"]]


# demo = gr.Interface(
#     fn=answer,
#     # inputs=["text", "slider"],
#     inputs=[gr.Textbox(label="question", lines=5)],
#     # outputs=[gr.TextArea(label="answer", lines=5),
#     #          gr.JSON(label="urls", value=list)]
#     outputs=[gr.Markdown(label="answer"),
#              gr.JSON(label="urls", value=list)]
# )


# demo.launch(server_name="0.0.0.0")


import json
import gradio as gr
from threading import Thread
import requests
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = "/root/models/Qwen1.5-1.8B-Chat"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# prompt
# prompt_template = """
# 你是一个云技术专家
# 使用以下检索到的Context回答问题。
# 如果不知道答案，就说不知道。
# 用中文回答问题。
# Question: {question}
# Context: {context}
# Answer:
# """

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
# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=prompt_template,
# )


def get_context_list(q: str):
    url = "http://10.0.0.7:8000/retriever"
    payload = {"content": q}
    res = requests.post(url, json=payload)
    return res.text


def question_answer(query):
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
                             max_new_tokens=200, do_sample=True, temperature=0.2)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generate_text = ''
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
