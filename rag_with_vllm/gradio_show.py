import json
import gradio as gr
import requests


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


def answer(question):
    url = "http://127.0.0.1:8888/answer"
    payload = {"content": question}
    res = requests.post(url, json=payload)
    res_json = json.loads(res.text)
    return [res_json["answer"], res_json["sources"]]


demo = gr.Interface(
    fn=answer,
    # inputs=["text", "slider"],
    inputs=[gr.Textbox(label="question", lines=5)],
    # outputs=[gr.TextArea(label="answer", lines=5),
    #          gr.JSON(label="urls", value=list)]
    outputs=[gr.Markdown(label="answer"),
             gr.JSON(label="urls", value=list)]
)


demo.launch(server_name="0.0.0.0")
