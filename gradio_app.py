import gradio as gr
import uvicorn
from app.main import app as fastapi_app
from app.inference import predict


# Funzione wrapper per Gradio
def gradio_predict(text):
    result = predict(text)
    return f"Label: {result['label']}\nConfidence: {result['confidence']:.2f}"


# Gradio interface
iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(lines=2, placeholder="Scrivi qui il testo"),
    outputs="text",
    title="Sentiment Analysis",
    description="Inserisci un testo e ottieni predizione di sentiment.",
)

if __name__ == "__main__":
    # Launch Gradio e FastAPI insieme
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # opzionale, crea link pubblico temporaneo
        inline=False,
    )
