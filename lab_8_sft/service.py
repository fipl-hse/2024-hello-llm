"""
Web service for model inference.
"""
# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code
from fastapi import FastAPI
import logging
from pathlib import Path
from config.lab_settings import LabSettings
from dataclasses import dataclass
import pandas as pd
from fastapi.staticfiles import StaticFiles
from lab_8_sft.main import LLMPipeline, TaskDataset
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
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
    question: str
    use_base_model: bool


@app.post("/infer")
async def infer(query: Query):
    logging.debug(f"Received query: {query.question}")
    inquiry = tuple([query.question])
    print(inquiry)
    if query.use_base_model:
        prediction = pre_trained_pipeline.infer_sample(inquiry)
    else:
        prediction = fine_tuned_pipeline.infer_sample(inquiry)
    logging.debug(f"Prediction: {prediction}")
    print(prediction, type(prediction))
    return {"infer": "Negative" if prediction == '0' else "Positive"}

    # print(query.question)
    # inquiry = tuple(query.question)
    # if query.use_base_model:
    #     prediction = pre_trained_pipeline.infer_sample(inquiry)
    # else:
    #     prediction = fine_tuned_pipeline.infer_sample(inquiry)
    # return {"infer": "Negative" if prediction == 0 else "Positive"}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint for the web application.
    Args:
        request (Request): The incoming HTTP request.
    Returns:
        HTMLResponse: The rendered HTML response for 'index.html'.
    """
    templates = Jinja2Templates(directory=Path(__file__).parent / "assets")
    return templates.TemplateResponse("index.html", {"request": request})
