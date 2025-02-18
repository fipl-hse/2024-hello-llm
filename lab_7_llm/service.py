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
from pydantic import BaseModel

from config.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, TaskDataset

ASSETS_PATH = Path(__file__).parent / "assets"


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    my_app = FastAPI()
    lab_settings = LabSettings(Path(__file__).parent / "settings.json")

    dataset = TaskDataset(pd.DataFrame())
    model_pipeline = LLMPipeline(
        model_name=lab_settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device="cpu"
    )
    return my_app, model_pipeline


app, pipeline = init_application()
app.mount("/static", StaticFiles(directory=ASSETS_PATH), name="static")
templates = Jinja2Templates(directory=str(ASSETS_PATH))


class Text(BaseModel):
    """
    Class for user's input
    """
    question: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint serving the main HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/infer")
async def infer(text: Text) -> JSONResponse:
    """
    Endpoint for model inference.
    """
    summarized_text = pipeline.infer_sample((text.question,))
    return JSONResponse(content={"infer": summarized_text})
