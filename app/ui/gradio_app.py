import gradio as gr
import requests

API_URL = "http://localhost:8000/predict"


def analyze(text: str):
    r = requests.post(API_URL, json={"text": text}, timeout=10)
    r.raise_for_status()
    return r.json()


def launch_gradio():
    demo = gr.Interface(
        fn=analyze,
        inputs=gr.Textbox(
            lines=4,
            label="Input text",
            placeholder="Type a sentence to analyze sentiment",
        ),
        outputs=gr.JSON(label="Prediction"),
        title="Sentiment Analysis API",
        description="FastAPI + FastText model served on Hugging Face Spaces",
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
