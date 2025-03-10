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
from pydantic.dataclasses import dataclass

from config.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, TaskDataset

PARENT_DIRECTORY = Path(__file__).parent


@dataclass
class Query:
    """
    A class to represent a query with a question.

    Attributes:
        question (str): User input text
    """
    question: str


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    settings = LabSettings(PARENT_DIRECTORY / "settings.json")

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
        app=StaticFiles(directory=PARENT_DIRECTORY / "assets"),
        name="assets"
    )

    return llm_app, llm_pipe


app, pipeline = init_application()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint for the web application.

    Args:
        request (Request): The incoming HTTP request object.

    Returns:
        HTMLResponse: The rendered HTML response containing the content of
        'index.html'.
    """
    templates = Jinja2Templates(directory=PARENT_DIRECTORY / "assets")
    return templates.TemplateResponse('index.html', {'request': request})


@app.post("/infer")
async def infer(query: Query) -> dict[str, str]:
    """
    Perform inference based on the provided query.

    Args:
        query (Query): Input query for inference.

    Returns:
        dict[str, str]: Inference results as a dictionary.
    """
    return {"infer": pipeline.infer_sample((query.question,))}
