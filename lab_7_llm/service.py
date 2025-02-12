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
    batch_size = 1
    device = 'cpu'

    summarization_pipeline = LLMPipeline(
        parameters.model, TaskDataset(pd.DataFrame()),
        max_length, batch_size, device
    )

    summarization_app = FastAPI()

    return summarization_app, summarization_pipeline


app, pipeline = init_application()

app_path = PROJECT_ROOT / 'lab_7_llm' / 'assets'
app.mount('/assets', StaticFiles(directory=app_path), name='assets')


@app.get('/', response_class=HTMLResponse)
async def read_root(request: Request):
    templates = Jinja2Templates(directory=app_path)
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/infer')
async def infer(request: Query):
    result = pipeline.infer_sample((request.question, ))
    return {'result': result}


