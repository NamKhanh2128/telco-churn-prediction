/**
 * Telco Churn Predictor — Frontend Logic
 *
 * Handles:
 * - Form submission → API call
 * - Result display (probability, risk, factors, recommendation)
 * - Error handling
 * - Loading states
 */

const API_BASE = window.location.origin;

// ─── DOM Elements ───────────────────────────────────
const form = document.getElementById("predict-form");
const btnPredict = document.getElementById("btn-predict");
const btnText = btnPredict.querySelector(".btn-text");
const btnLoading = btnPredict.querySelector(".btn-loading");

const resultSection = document.getElementById("result-section");
const probabilityValue = document.getElementById("probability-value");
const riskBadge = document.getElementById("risk-badge");
const riskText = document.getElementById("risk-text");
const predictionText = document.getElementById("prediction-text");
const factorsList = document.getElementById("factors-list");
const recommendationText = document.getElementById("recommendation-text");
const thresholdValue = document.getElementById("threshold-value");

const errorBanner = document.getElementById("error-banner");
const errorText = document.getElementById("error-text");

// ─── Form Submit ────────────────────────────────────
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    await predict();
});

async function predict() {
    // Collect form data
    const formData = new FormData(form);
    const data = {};

    for (const [key, value] of formData.entries()) {
        // Convert numeric fields
        if (["tenure", "SeniorCitizen"].includes(key)) {
            data[key] = parseInt(value, 10);
        } else if (["MonthlyCharges", "TotalCharges"].includes(key)) {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    }

    // UI: loading state
    setLoading(true);
    hideError();
    resultSection.style.display = "none";

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server error: ${response.status}`);
        }

        const result = await response.json();
        displayResult(result);
    } catch (err) {
        showError(err.message || "Failed to connect to API. Is the server running?");
    } finally {
        setLoading(false);
    }
}

// ─── Display Result ─────────────────────────────────
function displayResult(result) {
    // Probability
    const pct = (result.churn_probability * 100).toFixed(1);
    probabilityValue.textContent = `${pct}%`;

    // Color based on risk
    const riskColors = {
        Low: "var(--risk-low)",
        Medium: "var(--risk-medium)",
        High: "var(--risk-high)",
    };
    probabilityValue.style.color = riskColors[result.risk_level] || "var(--text-primary)";

    // Risk badge
    riskBadge.className = `risk-badge risk-${result.risk_level.toLowerCase()}`;
    riskText.textContent = `${result.risk_level} Risk`;

    // Prediction
    predictionText.textContent = `Prediction: ${result.prediction}`;

    // Top factors
    factorsList.innerHTML = "";
    if (result.top_factors && result.top_factors.length > 0) {
        result.top_factors.forEach((factor) => {
            const item = document.createElement("div");
            item.className = `factor-item impact-${factor.impact}`;
            item.innerHTML = `
                <span class="factor-name">${formatFeatureName(factor.feature)}</span>
                <span class="factor-value">${factor.value}</span>
                <span class="factor-impact impact-${factor.impact}">${factor.impact}</span>
            `;
            factorsList.appendChild(item);
        });
    }

    // Recommendation
    recommendationText.textContent = result.recommendation;

    // Threshold
    thresholdValue.textContent = result.threshold.toFixed(3);

    // Show result section
    resultSection.style.display = "block";

    // Scroll to result
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ─── Helpers ────────────────────────────────────────
function formatFeatureName(name) {
    // CamelCase / snake_case → human readable
    const map = {
        tenure: "Tenure",
        MonthlyCharges: "Monthly Charges",
        TotalCharges: "Total Charges",
        SeniorCitizen: "Senior Citizen",
        PhoneService: "Phone Service",
        MultipleLines: "Multiple Lines",
        InternetService: "Internet Service",
        OnlineSecurity: "Online Security",
        OnlineBackup: "Online Backup",
        DeviceProtection: "Device Protection",
        TechSupport: "Tech Support",
        StreamingTV: "Streaming TV",
        StreamingMovies: "Streaming Movies",
        PaperlessBilling: "Paperless Billing",
        PaymentMethod: "Payment Method",
        Contract: "Contract",
        Partner: "Partner",
        Dependents: "Dependents",
        gender: "Gender",
    };
    return map[name] || name;
}

function setLoading(isLoading) {
    btnPredict.disabled = isLoading;
    btnText.style.display = isLoading ? "none" : "inline";
    btnLoading.style.display = isLoading ? "inline" : "none";
}

function showError(message) {
    errorText.textContent = message;
    errorBanner.style.display = "block";
}

function hideError() {
    errorBanner.style.display = "none";
}
