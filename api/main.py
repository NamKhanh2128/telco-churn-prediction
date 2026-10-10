"""
FastAPI application — Telco Customer Churn Prediction API.

Endpoints:
    GET  /health            → Health check + model status
    POST /predict           → Predict churn for a single customer
    GET  /                  → Serve frontend

Run:
    uvicorn api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.schemas import CustomerInput, PredictionOutput, HealthResponse
from api.predict import load_model, predict_churn, get_threshold, is_model_loaded


# ─── Lifespan: load model on startup ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model on startup."""
    try:
        load_model()
        print("[api] Model loaded successfully")
    except FileNotFoundError as e:
        print(f"[api] WARNING: {e}")
        print("[api] API will start but /predict will return 503")
    yield


# ─── App ─────────────────────────────────────────────────
app = FastAPI(
    title="Telco Customer Churn Prediction",
    description=(
        "A machine-learning prototype for customer churn risk prediction "
        "based on the IBM Telco Customer Churn dataset. "
        "Returns probability, risk level, top factors, and recommendation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── Serve Frontend ─────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ─── Routes ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend index.html."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Telco Churn API. Visit /docs for API documentation."}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — verifies model is loaded."""
    return HealthResponse(
        status="ok" if is_model_loaded() else "model_not_loaded",
        model_loaded=is_model_loaded(),
        threshold=get_threshold(),
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(customer: CustomerInput):
    """
    Predict customer churn probability.

    Returns probability, risk level, top contributing factors,
    and an actionable recommendation.
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python -m src.train' first.",
        )

    try:
        result = predict_churn(customer.model_dump())
        return PredictionOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
