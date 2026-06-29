import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional

# ── Cargar artefactos ────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def load(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise RuntimeError(f"Artefacto no encontrado: {path}. Ejecuta train.py primero.")
    return joblib.load(path)

kbd_age       = load("kbd_age.pkl")
rob           = load("robust_scaler.pkl")
ohe           = load("ohe.pkl")
pipe_lr       = load("logistic_regression.pkl")
pipe_tree     = load("decision_tree.pkl")
pipe_knn      = load("knn.pkl")
feature_names = load("feature_names.pkl")
saved_metrics = load("metrics.pkl")

PIPELINES = {
    "logistic_regression": pipe_lr,
    "decision_tree":       pipe_tree,
    "knn":                 pipe_knn,
}

OHE_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Income Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema entrada ───────────────────────────────────────────────────────────
class PersonInput(BaseModel):
    age: int = Field(..., ge=17, le=90, description="Edad")
    workclass: Literal[
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ]
    education: Literal[
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
        "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
    ]
    education_num: int = Field(..., ge=1, le=16, alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: Literal[
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
    ]
    relationship: Literal[
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
    ]
    race: Literal["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Male", "Female"]
    capital_gain: int = Field(0, ge=0, alias="capital-gain")
    capital_loss: int = Field(0, ge=0, alias="capital-loss")
    hours_per_week: int = Field(..., ge=1, le=99, alias="hours-per-week")
    native_country: str = Field("United-States", alias="native-country")

    model_config = {"populate_by_name": True}


def build_feature_vector(p: PersonInput) -> np.ndarray:
    import pandas as pd

    age_bin = int(kbd_age.transform([[p.age]])[0, 0])
    cap_gain_log = np.log1p(p.capital_gain)
    cap_loss_log = np.log1p(p.capital_loss)
    hw_robust = float(rob.transform([[p.hours_per_week]])[0, 0])

    cat_data = pd.DataFrame([{
        "workclass":       p.workclass,
        "education":       p.education,
        "marital-status":  p.marital_status,
        "occupation":      p.occupation,
        "relationship":    p.relationship,
        "race":            p.race,
        "sex":             p.sex,
        "native-country":  p.native_country,
    }])
    ohe_vec = ohe.transform(cat_data)[0]

    row = np.concatenate([
        [age_bin],
        [p.age, cap_gain_log, cap_loss_log],
        ohe_vec,
        [p.education_num, hw_robust],
    ])

    import pandas as pd
    return pd.DataFrame([row], columns=feature_names)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_models():
    return {
        "models": [
            {"id": "logistic_regression", "name": "Regresión Logística"},
            {"id": "decision_tree",       "name": "Árbol de Decisión"},
            {"id": "knn",                 "name": "KNN"},
        ],
        "metrics": saved_metrics,
    }


@app.post("/predict")
def predict(person: PersonInput, model: Optional[str] = None):
    X = build_feature_vector(person)

    targets = {model: PIPELINES[model]} if model and model in PIPELINES else PIPELINES

    results = []
    for model_id, pipe in targets.items():
        prob = float(pipe.predict_proba(X)[0, 1])
        pred = int(pipe.predict(X)[0])
        results.append({
            "model_id":    model_id,
            "model_name":  next(m["name"] for m in [
                {"id": "logistic_regression", "name": "Regresión Logística"},
                {"id": "decision_tree",       "name": "Árbol de Decisión"},
                {"id": "knn",                 "name": "KNN"},
            ] if m["id"] == model_id),
            "prediction":  pred,
            "label":       ">50K" if pred == 1 else "≤50K",
            "probability_gt50k": round(prob, 4),
            "probability_le50k": round(1 - prob, 4),
        })

    return {"results": results}
