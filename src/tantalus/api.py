"""FastAPI endpoint: SMILES in, taste predictions + attribution heatmap out.

Run locally:
    uv run uvicorn tantalus.api:app --reload

Then ``POST /predict`` with ``{"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"}``.
The heatmap comes back base64-encoded inside the JSON, so one request gives
the client everything it needs to render a result. The response also reports
an applicability-domain check: the nearest training molecule by Tanimoto
similarity and whether the query is close enough for the prediction to be
trusted.
"""

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from tantalus.data import LABELS
from tantalus.predict import HEADLINE_LABELS, Ensemble

MAX_SMILES_LENGTH = 500  # keeps one request from running IG on a huge polymer


class PredictRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=MAX_SMILES_LENGTH)
    label: str | None = Field(None, description="label to explain; default: top one")


class LabelScore(BaseModel):
    probability: float
    seed_std: float
    validated: bool  # False for salty/sour: not scored on the scaffold split


class DomainInfo(BaseModel):
    """How close the query is to the training data (see ``tantalus.domain``)."""

    nearest_similarity: float
    nearest_smiles: str
    in_domain: bool


class PredictResponse(BaseModel):
    canonical_smiles: str
    model: str
    predictions: dict[str, LabelScore]
    explained_label: str
    attribution_png_base64: str
    attribution_delta: float
    applicability_domain: DomainInfo


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the checkpoints once at startup, not on every request."""
    app.state.ensemble = Ensemble()
    yield


app = FastAPI(title="Tantalus", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Predict every taste label and explain one with integrated gradients.

    Plain ``def`` (not ``async``): inference is CPU-bound, so FastAPI runs it
    in a worker thread instead of blocking the event loop.
    """
    if body.label is not None and body.label not in LABELS:
        raise HTTPException(422, f"unknown label {body.label!r}; expected {LABELS}")
    ensemble: Ensemble = request.app.state.ensemble
    result = ensemble.predict(body.smiles, body.label)
    if result is None:
        raise HTTPException(422, f"could not parse SMILES {body.smiles!r}")
    return PredictResponse(
        canonical_smiles=result.canonical_smiles,
        model=ensemble.model_name,
        predictions={
            label: LabelScore(
                probability=result.mean[label],
                seed_std=result.std[label],
                validated=label in HEADLINE_LABELS,
            )
            for label in LABELS
        },
        explained_label=result.explained_label,
        attribution_png_base64=base64.b64encode(result.attribution_png).decode(),
        attribution_delta=result.attribution_delta,
        applicability_domain=DomainInfo(
            nearest_similarity=result.domain.nearest_similarity,
            nearest_smiles=result.domain.nearest_smiles,
            in_domain=result.domain.in_domain,
        ),
    )
