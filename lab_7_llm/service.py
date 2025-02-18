"""
Web service for model inference.
"""
# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic.dataclasses import dataclass

from config.lab_settings import LabSettings
from config.constants import PROJECT_ROOT
from lab_7_llm.main import LLMPipeline, TaskDataset


@dataclass
class Query:
    """
    Make json request into a string
    """
    question: str


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    settings = LabSettings(PROJECT_ROOT /'lab_7_llm'/ "settings.json")

    llm_pipe = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=TaskDataset(pd.DataFrame()),
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    sum_app = FastAPI()

    sum_app.mount(
        path="/assets",
        app=StaticFiles(directory=PROJECT_ROOT /'lab_7_llm'/ "assets"),
        name="assets"
    )

    return sum_app, llm_pipe


app, pipeline = init_application()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint for the web application.
    """
    templates = Jinja2Templates(directory=PROJECT_ROOT /'lab_7_llm'/ "assets")
    return templates.TemplateResponse('index.html', {'request': request})


@app.post("/infer")
async def infer(text: Query) -> JSONResponse:
    """
    Endpoint for model inference.
    """
    summarized_text = pipeline.infer_sample((text.question,))
    return JSONResponse(content={"infer": summarized_text})
