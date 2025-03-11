"""
Web service for model inference.
"""
# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic.dataclasses import dataclass

from config.lab_settings import LabSettings
from lab_8_sft.main import LLMPipeline, TaskDataset
from lab_8_sft.start import main

LAB_PATH = Path(__file__).parent


def init_application() -> tuple[FastAPI, LLMPipeline, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn lab_8_sft.service:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline, LLMPipeline]: instance of server and pipeline
    """
    my_app = FastAPI()
    lab_settings = LabSettings(LAB_PATH / "settings.json")
    ft_model_path = LAB_PATH / "dist" / lab_settings.parameters.model
    if not ft_model_path.exists():
        main()

    dataset = TaskDataset(pd.DataFrame())
    original_pipeline = LLMPipeline(
        model_name=lab_settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    ft_pipeline = LLMPipeline(
        model_name=str(ft_model_path),
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device="cpu"
    )
    return my_app, original_pipeline, ft_pipeline


app, pre_trained_pipeline, fine_tuned_pipeline = init_application()

app.mount("/assets", StaticFiles(directory=LAB_PATH / "assets"), name="assets")
templates = Jinja2Templates(directory=str(LAB_PATH / "assets"))


@dataclass
class Query:
    """
    Class for user's input
    """
    question: str
    is_base_model: bool


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint serving the main HTML page.
    """
    return templates.TemplateResponse('index.html', {'request': request})


@app.post("/infer")
async def infer(text: Query) -> JSONResponse:
    """
    Endpoint for model inference.
    """
    if text.is_base_model:
        prediction = pre_trained_pipeline.infer_sample((text.question,))
    else:
        prediction = fine_tuned_pipeline.infer_sample((text.question,))
    return JSONResponse(content={"infer": prediction})
