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

PARENT_DIR = Path(__file__).resolve().parent
ASSETS_PATH = PARENT_DIR / "assets"


@dataclass
class Query:
    """
    Query model with question text
    """
    question: str
    is_base_model: bool


def init_application() -> tuple[FastAPI, LLMPipeline, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    settings = LabSettings(PARENT_DIR / "settings.json")

    application = FastAPI()
    application.mount("/static", StaticFiles(directory=str(ASSETS_PATH)), name="static")

    pre_trained_pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=TaskDataset(pd.DataFrame()),
        batch_size=1,
        max_length=120,
        device="cpu"
    )

    fine_tuned_model_path = PARENT_DIR / "dist" / settings.parameters.model
    if not fine_tuned_model_path.exists():
        main()

    fine_tuned_pipeline = LLMPipeline(
        model_name=str(fine_tuned_model_path),
        dataset=TaskDataset(pd.DataFrame()),
        batch_size=1,
        max_length=120,
        device="cpu"
    )

    return application, pre_trained_pipeline, fine_tuned_pipeline


app, pretrained_pipeline, finetuned_pipeline = init_application()

templates = Jinja2Templates(directory=str(ASSETS_PATH))


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    The root endpoint for rendering the start page
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/infer")
async def infer(query: Query) -> dict[str, str]:
    """
    The main endpoint for text processing via the LLM Pipeline
    """
    if query.is_base_model:
        return {"infer": pretrained_pipeline.infer_sample((query.question,))}
    return {"infer": finetuned_pipeline.infer_sample((query.question,))}
