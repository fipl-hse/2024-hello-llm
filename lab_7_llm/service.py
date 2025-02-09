"""
Web service for model inference.
"""
# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from lab_7_llm.main import LLMPipeline, TaskDataset


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """

    app = FastAPI()
    server_path = PROJECT_ROOT / 'assets'
    app.mount('/assets', StaticFiles(directory=server_path), name='assets')
    templates = Jinja2Templates(directory=server_path)

    @app.get('/', response_class=HTMLResponse)
    async def read_root(request: Request):
        return templates.TemplateResponse("index.html", {"request": request, "title": "Welcome to FastAPI",
                                                         "message": "This is a FastAPI service with static files."})

    @app.get("/assets/example")
    async def read_example_file():
        with open("reference_service/assets/example.txt", "r") as file:
            content = file.read()
        return {"content": content}



    @app.get('/items/{item_id}')
    async def read_item(item_id: int, query: str | None = None):
        return {'item_id': item_id, 'query': query}


    @app.get('/summarize')
    def predict_sentiment(text: str):
        settings_path = PROJECT_ROOT / 'lab_7_llm' / 'settings.json'
        parameters = LabSettings(settings_path).parameters

        max_length = 120
        batch_size = 64
        device = 'cpu'

        service_pipeline = LLMPipeline(
            parameters.model, None,
            max_length, batch_size, device
        )
        prediction = service_pipeline.infer_sample((text))

        return app, service_pipeline

app, pipeline = init_application()
