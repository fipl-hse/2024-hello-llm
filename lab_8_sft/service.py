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
from lab_8_sft.main import LLMPipeline, SFTPipeline, TaskDataset


def init_application() -> tuple[FastAPI, LLMPipeline, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    settings = LabSettings(Path(__file__).parent / 'settings.json')
    dataset = TaskDataset(pd.DataFrame())
    llm_pipeline = LLMPipeline(settings.parameters.model,
                               dataset, max_length=120, batch_size=1, device='cpu')

    finetuned_model_path = Path(__file__).parent / 'dist' / settings.parameters.model

    finetuned_pipeline = LLMPipeline(str(finetuned_model_path),
                               dataset, max_length=120, batch_size=1, device="cpu")
    fastapi = FastAPI()

    return fastapi, llm_pipeline, finetuned_pipeline


app, pipeline, finetuned_pipeline = init_application()

app.mount("/assets", StaticFiles(directory=Path(__file__).parent / "assets"), "assets")


@app.get("/")
async def root(request: Request) -> HTMLResponse:
    """
    Root  endpoint of the service
    """
    templates = Jinja2Templates(directory=Path(__file__).parent / "assets")
    return templates.TemplateResponse("index.html", {"request": request})

@dataclass
class Query:
    """
    A class for the question text
    """
    question: str
    is_base_model: bool


@app.post("/infer")
async def infer(query: Query) -> dict:
    """
    Main endpoint for model call
    """
    if query.is_base_model:
        response_text = pipeline.infer_sample((query.question,))
    else:
        response_text = finetuned_pipeline.infer_sample((query.question,))

    return {"infer": response_text}
