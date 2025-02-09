"""
Web service for model inference.
"""
# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code

from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from lab_7_llm.main import LLMPipeline, TaskDataset

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings

@dataclass
class SummarizedText:
    """Class representing a summarization task result."""
    summary: str


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """

    settings_path = PROJECT_ROOT / 'lab_7_llm' / 'settings.json'
    parameters = LabSettings(settings_path).parameters

    max_length = 120
    batch_size = 64
    device = 'cpu'

    service_pipeline = LLMPipeline(
        parameters.model, TaskDataset(pd.DataFrame()),
        max_length, batch_size, device
    )
    prediction = service_pipeline.infer_sample((text))


    app = FastAPI()

    server_path = PROJECT_ROOT / 'lab_7_llm' / 'assets'
    app.mount(server_path, StaticFiles(directory=server_path), name='assets')
    templates = Jinja2Templates(directory=server_path)

    @app.get('/', response_class=HTMLResponse)
    async def read_root(request: Request):
        return templates.TemplateResponse('index.html', {'request': request,
                                                         'title': 'test summarization',
                                                         'message': 'you can summarize any text you want here!'})


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
