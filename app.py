"""Inference-only API. Training runs separately with python main.py."""
import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, constr

from src.textSummarizer.pipeline.predicition_pipeline import PredictionPipeline

logger = logging.getLogger(__name__)
app = FastAPI(title="Conversational Text Summarizer")


class SummaryRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=1, max_length=20000)


class SummaryResponse(BaseModel):
    summary: str


@lru_cache(maxsize=1)
def get_predictor():
    return PredictionPipeline()


def predictor_dependency():
    try:
        return get_predictor()
    except Exception:
        logger.exception("Unable to load summarization checkpoint")
        raise HTTPException(status_code=503, detail="Model unavailable; check server logs.")


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=SummaryResponse)
def predict_route(request: SummaryRequest, predictor=Depends(predictor_dependency)):
    try:
        return SummaryResponse(summary=predictor.predict(request.text))
    except Exception:
        logger.exception("Summary generation failed")
        raise HTTPException(status_code=500, detail="Summary generation failed.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
