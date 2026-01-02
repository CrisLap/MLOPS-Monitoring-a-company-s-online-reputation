import gradio as gr
from app import inference

def gr_predict(text):
    result = inference.predict(text)
    return result["label"], result["confidence"]

demo = gr.Interface(
    fn=gr_predict,
    inputs=gr.Textbox(lines=3, placeholder="Inserisci testo qui..."),
    outputs=[
        gr.Textbox(label="Predicted Label"),
        gr.Number(label="Confidence"),
    ],
    title="Sentiment Analysis Demo",
    description="Demo interattiva del modello di sentiment"
)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

