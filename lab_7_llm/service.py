"""
Web service for model inference.
"""
# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code

from dataclasses import dataclass

import pandas as pd

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from lab_7_llm.main import LLMPipeline, TaskDataset

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings

@dataclass
class TextToSummarize:
    """Class representing an input text to be summarized."""
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

    summ_app = FastAPI()

    service_path = PROJECT_ROOT / 'lab_7_llm' / 'assets'
    summ_app.mount('/assets', StaticFiles(directory=service_path), name='assets')

    @summ_app.get('/', response_class=HTMLResponse)
    async def read_root():
        with open(service_path / 'index.html', 'r') as f:
            return HTMLResponse(content=f.read())


    @summ_app.post('/summarize')
    async def summarize(request: TextToSummarize):
        result = service_pipeline.infer_sample((request))
        return {'result': result}

    return summ_app, service_pipeline

app, pipeline = init_application()
