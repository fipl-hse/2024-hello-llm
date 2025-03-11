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


class InferenceRequest(BaseModel):
    """
    Schema for the incoming request to the inference endpoint.

    Attributes:
        use_finetuned (bool): Whether to use the fine-tuned model or not.
        text (str): The text to be passed for inference.
    """
    use_finetuned: bool
    text: str


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

    finetuned_pipeline = LLMPipeline(
        model_name=str(Path(__file__).parent / "dist" / lab_settings.parameters.model),
        dataset=TaskDataset(pd.DataFrame()),
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    return app_lmm, pretrained_pipeline, finetuned_pipeline


app, pre_trained_pipeline, fine_tuned_pipeline = init_application()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Get request to root, returns the index.html template.
    """
    templates = Jinja2Templates(directory=Path(__file__).parent / "assets")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/inference")
async def infer(request: InferenceRequest):
    """
    Handle inference requests and return predictions using the selected model.

    Args:
        request (InferenceRequest): Contains the input text and model choice.

    Returns:
        dict: Prediction result from the selected model.
    """
    if request.use_finetuned:
        pipeline = fine_tuned_pipeline
    else:
        pipeline = pre_trained_pipeline

    prediction = pipeline.infer_sample((request.text,))

    return {"prediction": prediction}
