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

from config.lab_settings import LabSettings
from lab_8_sft.main import LLMPipeline, TaskDataset
from lab_8_sft.start import main


def init_application() -> tuple[FastAPI, LLMPipeline, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn lab_8_sft.service:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline, LLMPipeline]: instance of server and pipeline
    """
    settings = LabSettings(Path(__file__).parent / "settings.json")

    fine_tuned_model_path = Path(__file__).parent / "dist" / settings.parameters.model
    if not fine_tuned_model_path.exists():
        main()

    llm_app = FastAPI()

    pre_trained_pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=TaskDataset(pd.DataFrame()),
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    fine_tuned_pipeline = LLMPipeline(
        model_name=str(fine_tuned_model_path),
        dataset=TaskDataset(pd.DataFrame()),
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    llm_app.mount(
        path="/assets",
        app=StaticFiles(directory=Path(__file__).parent / "assets"),
        name="assets"
    )

    return llm_app, pre_trained_pipeline, fine_tuned_pipeline


app, pre_trained_pipeline, fine_tuned_pipeline = init_application()


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


class Query(BaseModel):
    question: str
    is_base_model: bool


@app.post("/infer", response_model=None)
async def infer(query: Query) -> dict[str, str]:
    """
    Runs model inference.
    Args:
        query (Query): User input.
    Returns:
        dict[str, str]: Inference result.
    """
    if query.is_base_model:
        answer = pre_trained_pipeline.infer_sample((query.question,))
    else:
        answer = fine_tuned_pipeline.infer_sample((query.question,))

    return {"infer": answer}