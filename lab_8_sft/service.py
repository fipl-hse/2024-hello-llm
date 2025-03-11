"""
Web service for model inference.
"""
from pathlib import Path
from pydantic import dataclasses

import pandas as pd

from lab_8_sft.main import LLMPipeline, TaskDataset
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from seminars.seminar_02_12_2025.try_fastapi import APP_FOLDER

# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
except ImportError:
    print('Library "fastapi" not installed. Failed to import.')
    FastAPI = None


def init_application() -> tuple[FastAPI, LLMPipeline, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn lab_8_sft.service:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline, LLMPipeline]: instance of server and pipeline
    """
    settings = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=f"{APP_FOLDER}/static"), name="static")

    base_pipeline = LLMPipeline(
        settings.parameters.model,
        TaskDataset(pd.DataFrame()),
        120,
        1,
        "cpu"
    )

    model_path = Path(__file__).parent / "dist"
    sft_pipeline = LLMPipeline(
        str(model_path),
        TaskDataset(pd.DataFrame()),
        120,
        64,
        "cpu"
    )

    return app, base_pipeline, sft_pipeline

app, pre_trained_pipeline, fine_tuned_pipeline = init_application()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Endpoint to demonstrate the case when no dynamic data is loaded.

        Args:
         request (Request): A Request

    Returns:
        HTMLResponse: A response
    """
    templates = Jinja2Templates(directory=f"{PROJECT_ROOT}/assets")
    return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/infer")
# async def handle_get_request(request: Request) -> HTMLResponse:
#     """
#     Endpoint to demonstrate the case when no dynamic data is loaded.
#
#         Args:
#          request (Request): A Request
#
#     Returns:
#         HTMLResponse: A response
#     """
#     templates = Jinja2Templates(directory=f"{APP_FOLDER}/templates")
#     return templates.TemplateResponse("index.html", {"request": request})
