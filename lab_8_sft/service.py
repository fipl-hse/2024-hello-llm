"""
Web service for model inference.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
    fastapi_app = FastAPI()
    settings_path = Path(__file__).parent / 'settings.json'
    settings = LabSettings(settings_path)

    # variables
    max_length = 120
    batch_size = 1
    device = 'cpu'

    dataset = TaskDataset(pd.DataFrame())
    llm_pipeline = LLMPipeline(settings.parameters.model,
                               dataset,
                               max_length,
                               batch_size,
                               device)

    finetuned_llm_path = Path(__file__).parent / 'dist' / settings.parameters.model

    if not finetuned_llm_path.exists():
        main()

    finetuned_llm_pipeline = LLMPipeline(str(finetuned_llm_path),
                                         dataset,
                                         max_length,
                                         batch_size,
                                         device)

    return fastapi_app, llm_pipeline, finetuned_llm_pipeline


app, pre_trained_pipeline, fine_tuned_pipeline = init_application()
app.mount("/assets", StaticFiles(directory=Path(__file__).parent / "assets"), "assets")


@dataclass
class Query:
    """
    Class for queries + checkbox for pretrained | base model
    """
    question: str
    use_base_model: bool


@app.post("/infer")
async def infer(query: Query) -> dict:
    """
    Handle a POST request to classify the sentiment of the input text.

    Args:
        query (Query): A dataclass containing:
            - question (str): The input text to classify.
            - use_base_model (bool): If True, use the base model;
            otherwise, use the fine-tuned model.

    Returns:
        dict: A dictionary containing the sentiment classification result:
            - "infer" (str): The sentiment classification, either "Negative" or "Positive".
    """
    logging.debug("Received query: %s", query.question)
    inquiry = tuple([query.question])
    print(inquiry)
    if query.use_base_model:
        prediction = pre_trained_pipeline.infer_sample(inquiry)
    else:
        prediction = fine_tuned_pipeline.infer_sample(inquiry)
    logging.debug("Prediction: %s", prediction)
    print(prediction, type(prediction))
    return {"infer": "Negative" if prediction == '0' else "Positive"}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Serve the main HTML page for the web application.

    Args:
        request (Request): The incoming HTTP request object, which is passed to the template
                           for rendering.

    Returns:
        HTMLResponse: A response containing the rendered HTML page from the 'index.html' template.
    """
    templates = Jinja2Templates(directory=Path(__file__).parent / "assets")
    return templates.TemplateResponse("index.html", {"request": request})
