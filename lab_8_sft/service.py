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
    Query model with question text and model selection flag.
    Attributes:
        question (str): User input text.
        is_base_model (bool): Flag to use either base or fine-tuned model.
    """
    question: str
    is_base_model: bool


def init_application() -> tuple[FastAPI, LLMPipeline, LLMPipeline]:
    """
    Initialize the FastAPI application and load both base and fine-tuned models.
    Returns:
        tuple[fastapi.FastAPI, LLMPipeline, LLMPipeline]: Server instance and two model pipelines.
    """
    settings = LabSettings(PARENT_DIR / "settings.json")

    # Создаем FastAPI приложение и монтируем статические файлы
    application = FastAPI()
    application.mount("/static", StaticFiles(directory=str(ASSETS_PATH)), name="static")

    # Загружаем предобученную модель
    pre_trained_pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=TaskDataset(pd.DataFrame()),
        batch_size=1,
        max_length=120,
        device="cpu"
    )

    # Проверяем, существует ли дообученная модель, если нет — запускаем main() для её обучения
    fine_tuned_model_path = PARENT_DIR / "dist" / settings.parameters.model
    if not fine_tuned_model_path.exists():
        main()

    # Загружаем дообученную модель
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
    Root endpoint for the web application.
    Returns:
        HTMLResponse: The rendered HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/infer")
async def infer(query: Query) -> dict[str, str]:
    """
    Perform inference based on the user query and selected model.
    Args:
        query (Query): User input and model selection flag.
    Returns:
        dict[str, str]: The model's inference result.
    """
    if query.is_base_model:
        return {"infer": pretrained_pipeline.infer_sample((query.question,))}
    return {"infer": finetuned_pipeline.infer_sample((query.question,))}
