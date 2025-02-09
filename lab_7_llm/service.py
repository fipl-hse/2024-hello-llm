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

BASE_DIR = Path(__file__).resolve().parent
ASSETS_PATH = BASE_DIR / "assets"


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    app = FastAPI()

    settings = LabSettings(BASE_DIR / "settings.json")
    dataset = TaskDataset(pd.DataFrame())
    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=64,
        device="cpu"
    )

    return app, pipeline


app, pipeline = init_application()
app.mount("/static", StaticFiles(directory=ASSETS_PATH), name="static")
templates = Jinja2Templates(directory=str(ASSETS_PATH))


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    root endpoint serving the main HTML page
    """
    return templates.TemplateResponse("index.html", {"request": request})


class Query(BaseModel):
    """
    data model for API request
    """
    question: str


@app.post("/infer")
async def infer(query: Query):
    """
    main endpoint for model inference
    """
    result = pipeline.infer_sample((query.question,))
    return JSONResponse(content={"infer": result})
