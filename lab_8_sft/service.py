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
from lab_8_sft.main import LLMPipeline, TaskDataset

PARENT_DIR = Path(__file__).parent
ASSETS_PATH = PARENT_DIR / 'assets'


def init_application() -> tuple[FastAPI, LLMPipeline, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn lab_8_sft.service:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline, LLMPipeline]: instance of server and pipeline
    """
    settings = LabSettings(PARENT_DIR / 'settings.json')

    model_app = FastAPI()
    model_app.mount('/assets', StaticFiles(directory=ASSETS_PATH), name='assets')

    pre_trained_llm = LLMPipeline(model_name=settings.parameters.model, max_length=120,
                                  dataset=TaskDataset(pd.DataFrame()), batch_size=1, device='cpu')

    fine_tuned_model_path = PARENT_DIR / 'dist' / settings.parameters.model

    fine_tuned_llm = LLMPipeline(model_name=str(fine_tuned_model_path), max_length=120,
                                 dataset=TaskDataset(pd.DataFrame()), batch_size=1, device='cpu')

    return model_app, pre_trained_llm, fine_tuned_llm


app, pre_trained_pipeline, fine_tuned_pipeline = init_application()
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
    hypothesis: str
    is_base_model: bool


@app.post("/infer")
def infer(query: Query) -> dict[str, str]:
    """
    Creates a main endpoint for model call.
    Args:
        query (Query): Passed data for HTML.
    Returns:
        dict[str, str]: Inference results as a dictionary.
    """
    label_mapping = {'0': 'entailment', '1': 'neutral', '2': 'contradiction'}
    sample = query.question, query.hypothesis
    if query.is_base_model:
        prediction = pre_trained_pipeline.infer_sample(sample)
        return {'infer': label_mapping.get(prediction)}
    prediction = fine_tuned_pipeline.infer_sample(sample)
    return {'infer': label_mapping.get(prediction)}
