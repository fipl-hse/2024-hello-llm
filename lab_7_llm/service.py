"""
Web service for model inference.
"""
# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code

from dataclasses import dataclass

import pandas as pd
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from lab_7_llm.main import LLMPipeline, TaskDataset

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings

@dataclass
class Query:
    """Class representing an input text to be summarized."""
    question: str


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

    summ_app = FastAPI()

    service_path = PROJECT_ROOT / 'lab_7_llm' / 'assets'
    summ_app.mount('/assets', StaticFiles(directory=service_path), name='assets')

    return summ_app, service_pipeline


    @summ_app.get('/', response_class=HTMLResponse)
    async def read_root(request: Request):
        templates = Jinja2Templates(directory=service_path)
        return templates.TemplateResponse('index.html', {'request': request})


    @summ_app.post('/infer')
    async def infer(request: Query):
        result = service_pipeline.infer_sample((Query.question, ))
        return {'result': result}


app, pipeline = init_application()