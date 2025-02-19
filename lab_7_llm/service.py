"""
Web service for model inference.
"""
# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code

import logging
from dataclasses import dataclass

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, TaskDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Query:
    """
    Class representing an input text to be summarized by the model.
    """
    question: str


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn lab_7_llm.service:app --reload

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

logger.info('fastapi application started')


@app.get('/', response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Serve the root endpoint of the service by rendering the main HTML template.

    Args:
        request (Request): The incoming HTTP request object, provided by FastAPI

    Returns:
        TemplateResponse: An HTMLResponse object containing the rendered `index.html` template
    """
    templates = Jinja2Templates(directory=app_path)
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/infer')
async def infer(request: Query) -> dict[str, str]:
    """
    Create an endpoint for model call.

    Args:
        request (Query): The incoming HTTP request object containing the input text

    Returns:
        dict[str, str]: A dictionary containing the inference results
    """
    logger.info('received request: %s', request.question)
    result = pipeline.infer_sample((request.question, ))
    logger.info('model inference complete: %s', result)

    return {'infer': result}
