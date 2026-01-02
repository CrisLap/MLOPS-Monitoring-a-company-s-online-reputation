import threading
import uvicorn
from ui.gradio_app import launch_gradio


def start_api():
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    # FastAPI in thread separato
    api_thread = threading.Thread(target=start_api)
    api_thread.start()

    # Gradio (porta standard HF)
    launch_gradio()
