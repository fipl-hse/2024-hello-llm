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
from lab_8_sft.main import LLMPipeline, TaskDataset
from lab_8_sft.start import main


class InferenceRequest(BaseModel):
    """
    Schema for the incoming request to the inference endpoint.

    Attributes:
        use_finetuned (bool): Whether to use the fine-tuned model or not.
        text (str): The text to be passed for inference.
    """
    is_base_model: bool
    question: str


def init_application() -> tuple[FastAPI, LLMPipeline, LLMPipeline]:
    """
    Initialize core application.


    Run: uvicorn lab_8_sft.service:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline, LLMPipeline]: instance of server and pipeline
    """
    app_lmm = FastAPI()

    app_lmm.mount("/assets",
                  StaticFiles(directory=Path(__file__).parent / "assets"),
                  name="static")

    lab_settings = LabSettings(PROJECT_ROOT / 'lab_8_sft' / 'settings.json')

    pretrained_pipeline = LLMPipeline(
        model_name=lab_settings.parameters.model,
        dataset=TaskDataset(pd.DataFrame()),
        max_length=120,
        batch_size=1,
        device="cpu"
    )
    fine_tuned_model_path = Path(__file__).parent / "dist" / lab_settings.parameters.model
    if not fine_tuned_model_path.exists():
        main()

    finetuned_pipeline = LLMPipeline(
        model_name=str(fine_tuned_model_path),
        dataset=TaskDataset(pd.DataFrame()),
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    return app_lmm, pretrained_pipeline, finetuned_pipeline


app, pre_trained_pipeline, fine_tuned_pipeline = init_application()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    """
    Get request to root, returns the index.html template.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        HTMLResponse: The rendered HTML response for 'index.html'.
    """
    templates = Jinja2Templates(directory=Path(__file__).parent / "assets")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/infer")
async def infer(request: InferenceRequest) -> dict[str, str]:
    """
    Handle inference requests and return predictions using the selected model.

    Args:
        request (InferenceRequest): Contains the input text and model choice.

    Returns:
        dict: Prediction result from the selected model.
    """
    if request.is_base_model:
        pipeline = pre_trained_pipeline
    else:
        pipeline = fine_tuned_pipeline

    return {"infer": pipeline.infer_sample((request.question,))}
