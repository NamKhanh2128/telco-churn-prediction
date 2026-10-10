"""
Pydantic schemas for API request/response.

Response includes:
- churn_probability
- prediction (Churn / No Churn)
- risk_level (Low / Medium / High)
- threshold (dynamic, from config.json)
- top_factors (aggregated SHAP → original feature names)
- recommendation
"""

from pydantic import BaseModel, Field
from typing import Optional


class CustomerInput(BaseModel):
    """Input: customer profile features."""

    gender: str = Field(..., examples=["Male", "Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes", "No"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No", "Yes", "No phone service"])
    InternetService: str = Field(..., examples=["Fiber optic", "DSL", "No"])
    OnlineSecurity: str = Field(..., examples=["No", "Yes", "No internet service"])
    OnlineBackup: str = Field(..., examples=["No", "Yes", "No internet service"])
    DeviceProtection: str = Field(..., examples=["No", "Yes", "No internet service"])
    TechSupport: str = Field(..., examples=["No", "Yes", "No internet service"])
    StreamingTV: str = Field(..., examples=["No", "Yes", "No internet service"])
    StreamingMovies: str = Field(..., examples=["No", "Yes", "No internet service"])
    Contract: str = Field(..., examples=["Month-to-month", "One year", "Two year"])
    PaperlessBilling: str = Field(..., examples=["Yes", "No"])
    PaymentMethod: str = Field(
        ...,
        examples=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    MonthlyCharges: float = Field(..., ge=0, examples=[75.5])
    TotalCharges: Optional[float] = Field(None, ge=0, examples=[906.0])


class ChurnFactor(BaseModel):
    """A single factor contributing to the churn prediction."""
    feature: str
    value: str
    impact: str  # "high", "medium", "low"


class PredictionOutput(BaseModel):
    """Output: churn prediction with explanation."""

    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn (0-1)")
    prediction: str = Field(..., description="Churn or No Churn")
    risk_level: str = Field(..., description="Low, Medium, or High")
    threshold: float = Field(..., description="Decision threshold used")
    top_factors: list[ChurnFactor] = Field(default_factory=list, description="Top factors driving the prediction")
    recommendation: str = Field(..., description="Actionable recommendation")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    threshold: float
