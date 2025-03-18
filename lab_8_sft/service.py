"""
Web service for model inference.
"""
# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code
from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic.dataclasses import dataclass

from config.lab_settings import LabSettings
from lab_8_sft.main import LLMPipeline, TaskDataset


@dataclass
class Query:
    """
    Class for query content.
    """
    question: str
    is_base_model: bool


def init_application() -> tuple[FastAPI, LLMPipeline, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn lab_8_sft.service:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline, LLMPipeline]: instance of server and pipeline
    """
    max_length = 120
    batch_size = 1
    device = 'cpu'
    settings_path = Path(__file__).parent / 'settings.json'
    settings = LabSettings(settings_path)
    # finetuned_model_path = Path(__file__).parent / 'dist' / f'{settings.parameters.model}-finetuned'
    finetuned_model_path = Path(__file__).parent / 'dist' / settings.parameters.model
    empty_dataset = TaskDataset(pd.DataFrame())

    fastapi_app = FastAPI()
    fastapi_app.mount('/static',
                      StaticFiles(directory=f'{Path(__file__).parent}/assets'), name='static')

    pre_trained = LLMPipeline(
        settings.parameters.model,
        empty_dataset,
        max_length,
        batch_size,
        device
    )

    fine_tuned = LLMPipeline(
        str(finetuned_model_path),
        empty_dataset,
        max_length,
        batch_size,
        device
    )

    return fastapi_app, pre_trained, fine_tuned

app, pre_trained_pipeline, fine_tuned_pipeline = init_application()

@app.get('/', response_class=HTMLResponse)
async def root(request: Request) -> Callable:
    """
    Root endpoint.

    Args:
        request: GET-request to the server

    Returns:
        HTML-template
    """
    templates = Jinja2Templates(directory=f'{Path(__file__).parent}/assets')
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/infer')
async def infer(query: Query) -> JSONResponse:
    """
    Main endpoint for model inference.

    Args:
        query: Query instance, containing text to classify

    Returns:
        JSONResponse instance in {"infer": prediction} format
    """
    if query.is_base_model:
        prediction = pre_trained_pipeline.infer_sample((query.question, ))
    else:
        prediction = fine_tuned_pipeline.infer_sample((query.question,))

    return JSONResponse(content={'infer': prediction})
