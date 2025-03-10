"""
Web service for model inference.
"""
import pandas as pd

# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config.lab_settings import LabSettings
from lab_8_sft.main import LLMPipeline


def init_application() -> tuple[FastAPI, LLMPipeline, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn lab_8_sft.service:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline, LLMPipeline]: instance of server and pipeline
    """


app, pre_trained_pipeline, fine_tuned_pipeline = (None, None, None)
