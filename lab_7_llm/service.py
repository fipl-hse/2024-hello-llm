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
from pydantic import BaseModel

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, TaskDataset

PARENT_DIRECTORY = Path(__file__).parent


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    llm_app = FastAPI(
        title="IT-EMOTION-ANALYZER"
    )

    llm_app.mount("/assets",
                  StaticFiles(directory=PARENT_DIRECTORY / "assets"),
                  name="assets")

    lab_settings = LabSettings(PROJECT_ROOT / 'lab_7_llm' / 'settings.json')
    llm_pipeline = LLMPipeline(
        model_name=lab_settings.parameters.model,
        dataset=TaskDataset(pd.DataFrame()),
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    llm_app.state.pipeline = llm_pipeline
    print("Pipeline Initialized")

    return llm_app, llm_pipeline


app, pipeline = init_application()


@app.get("/")
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint for the service.
    Returns:
        TemplateResponse: Renders the index.html page with Jinja2.
    """
    templates = Jinja2Templates(directory=PARENT_DIRECTORY / "assets")
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


class Query(BaseModel):
    """
    A class to represent a query containing a question.

    Attributes:
        question (str): The text of the user's input.
    """
    question: str


@app.post("/infer")
async def infer(query: Query) -> dict[str, str]:
    """
    Main endpoint for processing user queries using the LLM pipeline.
    Args:
        query (Query): User query.
    Returns:
        dict: Response containing the inference result.
    """
    input_text = query.question
    response = pipeline.infer_sample((input_text,))
    tags_to_text = {'0': 'sadness', '1': 'joy', '2': 'love', '3':
                    'anger', '4': 'fear', '5': 'surprise'}
    return {"infer": tags_to_text[response]}
