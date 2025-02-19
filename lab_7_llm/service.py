"""
Web service for model inference.
"""
# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dataclasses import dataclass

from config.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, TaskDataset

PARENT_DIRECTORY = Path(__file__).parent


@dataclass
class Query:
    """
    Represents a user query.
    """
    question: str


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    settings = LabSettings(Path(__file__).parent / "settings.json")
    llm_pipe = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=TaskDataset(pd.DataFrame()),
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    llm_app = FastAPI()

    llm_app.mount(
        path="/assets",
        app=StaticFiles(directory=Path(__file__).parent / "assets"),
        name="assets"
    )

    return llm_app, llm_pipe


app, pipeline = init_application()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Serves the main page.
    Args:
        request (Request): Incoming HTTP request.
    Returns:
        HTMLResponse: Rendered 'index.html'.
    """
    templates = Jinja2Templates(directory=Path(__file__).parent / "assets")
    return templates.TemplateResponse('index.html', {'request': request})


@app.post("/infer")
async def infer(query: Query) -> dict[str, str]:
    """
    Runs model inference.
    Args:
        query (Query): User input.
    Returns:
        dict[str, str]: Inference result.
    """
    answer = pipeline.infer_sample((query.question,))
    return {"infer": answer}
