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


class Query(BaseModel):
    question: str
    is_base_model: bool = True


def init_application() -> tuple[FastAPI, LLMPipeline, LLMPipeline]:
    """
    Initialize core application.

    Run: uvicorn lab_8_sft.service:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline, LLMPipeline]: instance of server and pipeline
    """
    settings = LabSettings(Path(__file__).parent / "settings.json")

    llm_app = FastAPI()
    llm_app.mount(
        "/assets",
        StaticFiles(directory=Path(__file__).parent / "assets"),
        "assets"
    )

    empty_dataset = TaskDataset(pd.DataFrame())

    base_llm_pipeline = LLMPipeline(settings.parameters.model,
                                    empty_dataset,
                                    max_length=120,
                                    batch_size=1,
                                    device="cpu"
                                    )

    fine_tuned_model_path = Path(__file__).parent / "dist" / settings.parameters.model
    if not fine_tuned_model_path.exists():
        main()

    fine_tuned_llm_pipeline = LLMPipeline(str(fine_tuned_model_path),
                                          empty_dataset,
                                          max_length=120,
                                          batch_size=1,
                                          device="cpu"
                                          )

    return llm_app, base_llm_pipeline, fine_tuned_llm_pipeline


app, pre_trained_pipeline, fine_tuned_pipeline = init_application()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    templates = Jinja2Templates(directory=Path(__file__).parent / "assets")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/infer")
async def infer(query: Query) -> dict[str, str]:
    q = query.question.strip()
    if not q:
        return {"infer": "Please provide a valid query."}

    if query.is_base_model:
        answer = pre_trained_pipeline.infer_sample((q,))
    else:
        answer = fine_tuned_pipeline.infer_sample((q,))

    return {"infer": answer}
