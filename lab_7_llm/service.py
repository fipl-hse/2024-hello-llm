"""
Web service for model inference.
"""
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic.dataclasses import dataclass

from config.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, TaskDataset

PARENT_DIR = Path(__file__).parent
ASSETS_PATH = PARENT_DIR / 'assets'


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    settings = LabSettings(PARENT_DIR / 'settings.json')

    model_app = FastAPI()
    model_app.mount('/assets', StaticFiles(directory=ASSETS_PATH), name='assets')

    model_pipeline = LLMPipeline(model_name=settings.parameters.model,
                                 dataset=TaskDataset(pd.DataFrame()), max_length=120, batch_size=1,
                                 device='cpu')
    return model_app, model_pipeline


app, pipeline = init_application()
templates = Jinja2Templates(directory=ASSETS_PATH)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Creates a root endpoint of the service.
    Args:
        request (Request): passed data for HTML.
    Returns:
        HTMLResponse: an HTML page
    """
    return templates.TemplateResponse('index.html', {'request': request})


@dataclass
class Query:
    """
    A class that contains text of the query.
    """
    question: str


@app.post("/infer")
def infer(query: Query):
    """
    Creates a main endpoint for model call.
    Args:
        query (Query): Passed data for HTML.
    Returns:
        dict: Response obtained as a result of the pipeline.
    """
    return {"infer": pipeline.infer_sample((query.question,))}
