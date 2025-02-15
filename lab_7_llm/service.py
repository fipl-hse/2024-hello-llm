"""
Web service for model inference.
"""
# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic.dataclasses import dataclass

from config.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, TaskDataset

PARENT_DIR = Path(__file__).resolve().parent
ASSETS_PATH = PARENT_DIR / "assets"


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    application = FastAPI()

    settings = LabSettings(PARENT_DIR / "settings.json")
    dataset = TaskDataset(pd.DataFrame())

    batch_size = 64
    max_length = 120
    device = 'cpu'
    model_pipeline = LLMPipeline(model_name=settings.parameters.model,
                                 dataset=dataset,
                                 batch_size=batch_size,
                                 max_length=max_length,
                                 device=device)

    return application, model_pipeline


app, pipeline = init_application()

app.mount("/static", StaticFiles(directory=str(ASSETS_PATH)), name="static")
templates = Jinja2Templates(directory=str(ASSETS_PATH))


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    The root endpoint for rendering the start page
    """
    return templates.TemplateResponse("index.html", {"request": request})


@dataclass
class Query:
    """
    Query model with question text
    """
    question: str


@app.post("/infer")
async def infer(query: Query) -> JSONResponse:
    """
    The main endpoint for text processing via the LLM Pipeline
    """
    response_text = pipeline.infer_sample((query.question,))
    return JSONResponse(content={"infer": response_text})
