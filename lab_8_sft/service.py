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
from lab_8_sft.main import LLMPipeline, TaskDataset
from lab_8_sft.start import main

LAB_FOLDER = Path(__file__).parent

@dataclass
class Query:
    """
    A class to represent a query with a question.

    Attributes:
        question (str): User input text
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
    settings = LabSettings(LAB_FOLDER / "settings.json")

    llm_app = FastAPI()
    llm_app.mount(
        path="/assets",
        app=StaticFiles(directory=LAB_FOLDER / "assets"),
        name="assets"
    )

    pre_trained_llm_pipe = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=TaskDataset(pd.DataFrame()),
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    fine_tuned_model_path = LAB_FOLDER / "dist" / settings.parameters.model
    if not fine_tuned_model_path.exists():
        main()

    fine_tuned_llm_pipe = LLMPipeline(
        model_name=str(fine_tuned_model_path),
        dataset=TaskDataset(pd.DataFrame()),
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    return llm_app, pre_trained_llm_pipe, fine_tuned_llm_pipe


app, pre_trained_pipeline, fine_tuned_pipeline = init_application()


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
    templates = Jinja2Templates(directory=LAB_FOLDER / "assets")
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
    if query.is_base_model:
        return {"infer": pre_trained_pipeline.infer_sample((query.question,))}
    return {"infer": fine_tuned_pipeline.infer_sample((query.question,))}
