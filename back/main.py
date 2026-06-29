import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def load(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise RuntimeError(f"Artefacto no encontrado: {path}. Ejecuta train.py primero.")
    return joblib.load(path)

kbd_age            = load("kbd_age.pkl")
rob                = load("robust_scaler.pkl")
ohe                = load("ohe.pkl")
pipe_lr            = load("logistic_regression.pkl")
pipe_tree          = load("decision_tree.pkl")
pipe_knn           = load("knn.pkl")
feature_names      = load("feature_names.pkl")
saved_metrics      = load("metrics.pkl")
roc_data           = load("roc_data.pkl")
cm_data            = load("cm_data.pkl")
feature_importance = load("feature_importance.pkl")
dataset_stats      = load("dataset_stats.pkl")

PIPELINES = {
    "logistic_regression": pipe_lr,
    "decision_tree":       pipe_tree,
    "knn":                 pipe_knn,
}

MODEL_NAMES = {
    "logistic_regression": "Regresión Logística",
    "decision_tree":       "Árbol de Decisión",
    "knn":                 "KNN",
}

app = FastAPI(title="Income Classifier API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PersonInput(BaseModel):
    age: int = Field(..., ge=17, le=90)
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

def build_feature_vector(p: PersonInput):
    import pandas as pd

    age_bin      = int(kbd_age.transform([[p.age]])[0, 0])
    cap_gain_log = float(np.log1p(p.capital_gain))
    cap_loss_log = float(np.log1p(p.capital_loss))
    hw_robust    = float(rob.transform([[p.hours_per_week]])[0, 0])

    cat_df = pd.DataFrame([{
        "workclass":      p.workclass,
        "education":      p.education,
        "marital-status": p.marital_status,
        "occupation":     p.occupation,
        "relationship":   p.relationship,
        "race":           p.race,
        "sex":            p.sex,
        "native-country": p.native_country,
    }])
    ohe_vec = ohe.transform(cat_df)[0]

    row = np.concatenate([
        [age_bin],
        [p.age, cap_gain_log, cap_loss_log],
        ohe_vec,
        [p.education_num, hw_robust],
    ])
    return pd.DataFrame([row], columns=feature_names)

PREFIX_LABELS = {
    "workclass":      "Sector laboral",
    "education":      "Nivel educativo",
    "marital-status": "Estado civil",
    "occupation":     "Ocupación",
    "relationship":   "Rol familiar",
    "race":           "Origen étnico",
    "sex":            "Sexo",
    "native-country": "País de origen",
}

VALUE_LABELS = {
    "Married-civ-spouse": "casado/a", "Never-married": "soltero/a",
    "Divorced": "divorciado/a", "Separated": "separado/a",
    "Widowed": "viudo/a", "Married-spouse-absent": "casado/a (cónyuge ausente)",
    "Husband": "esposo", "Wife": "esposa", "Own-child": "hijo/a dependiente",
    "Not-in-family": "sin familia a cargo", "Unmarried": "no casado/a",
    "Other-relative": "otro familiar",
    "Exec-managerial": "ejecutivo/gerencial", "Prof-specialty": "profesional especializado",
    "Tech-support": "soporte técnico", "Sales": "ventas", "Craft-repair": "oficios y reparación",
    "Adm-clerical": "administrativo", "Other-service": "servicios generales",
    "Machine-op-inspct": "operario de máquinas", "Handlers-cleaners": "manipulación y limpieza",
    "Transport-moving": "transporte", "Farming-fishing": "agricultura/pesca",
    "Protective-serv": "servicios de protección", "Priv-house-serv": "servicio doméstico",
    "Armed-Forces": "fuerzas armadas",
    "Male": "hombre", "Female": "mujer",
    "Private": "sector privado", "Self-emp-not-inc": "trabajador independiente",
    "Self-emp-inc": "empresario (con sociedad)", "Federal-gov": "gobierno federal",
    "Local-gov": "gobierno local", "State-gov": "gobierno estatal",
}

NUMERIC_LABELS = {
    "age":              "Edad",
    "age_entropia":     "Rango de edad",
    "capital-gain_log": "Ganancia de capital",
    "capital-loss_log": "Pérdida de capital",
    "education-num":    "Años de educación",
    "hours-per-week":   "Horas trabajadas por semana",
}

def _humanize(feature, person: PersonInput):
    if feature in NUMERIC_LABELS:
        detail = {
            "age":              f"{person.age} años",
            "age_entropia":     f"{person.age} años",
            "education-num":    f"{person.education_num} años",
            "capital-gain_log": f"${person.capital_gain:,}",
            "capital-loss_log": f"${person.capital_loss:,}",
            "hours-per-week":   f"{person.hours_per_week} h/semana",
        }.get(feature, "")
        return NUMERIC_LABELS[feature], detail

    for prefix, label in PREFIX_LABELS.items():
        if feature.startswith(prefix + "_"):
            raw_value = feature[len(prefix) + 1:]
            value = VALUE_LABELS.get(raw_value, raw_value)
            return label, value

    return feature, ""

def explain_prediction(X, person: PersonInput):
    lr = pipe_lr.named_steps["m"]
    sc = pipe_lr.named_steps["sc"]

    x        = X.iloc[0].values.astype(float)
    x_scaled = (x - sc.mean_) / sc.scale_
    coefs    = lr.coef_[0]
    contribs = coefs * x_scaled
    intercept = float(lr.intercept_[0])
    logit    = intercept + float(contribs.sum())
    prob     = 1.0 / (1.0 + np.exp(-logit))
    pred     = int(prob >= 0.5)

    factors = []
    for feat, c in zip(feature_names, contribs):
        if abs(c) < 1e-6:
            continue
        label, detail = _humanize(feat, person)
        factors.append({
            "feature":      feat,
            "label":        label,
            "detail":       detail,
            "contribution": round(float(c), 4),
            "direction":    "positivo" if c > 0 else "negativo",
        })

    factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)
    top = factors[:6]
    pos = [f for f in top if f["contribution"] > 0]
    neg = [f for f in top if f["contribution"] < 0]

    pct = round(prob * 100, 1)
    verdict = "supera los $50.000 anuales" if pred == 1 else "NO supera los $50.000 anuales"

    def _frase(fs):
        partes = [f"{f['label'].lower()} ({f['detail']})" if f["detail"] else f["label"].lower() for f in fs]
        if not partes:
            return ""
        if len(partes) == 1:
            return partes[0]
        return ", ".join(partes[:-1]) + " y " + partes[-1]

    narrativa = (
        f"El sistema estima una probabilidad del {pct}% de que esta persona "
        f"{verdict}. "
    )
    if pos:
        narrativa += f"Los factores que más empujaron hacia un ingreso ALTO fueron: {_frase(pos)}. "
    if neg:
        narrativa += f"En sentido contrario, los factores que reducen la probabilidad de ingreso alto fueron: {_frase(neg)}. "

    confianza = "alta" if abs(prob - 0.5) > 0.35 else "moderada" if abs(prob - 0.5) > 0.15 else "baja"
    narrativa += f"El nivel de confianza de esta decisión es {confianza}."

    return {
        "prediction":        pred,
        "label":             ">50K" if pred == 1 else "≤50K",
        "probability_gt50k": round(float(prob), 4),
        "confidence_level":  confianza,
        "narrative":         narrativa,
        "positive_factors":  pos,
        "negative_factors":  neg,
        "base_value":        round(intercept, 4),
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def get_models():
    return {
        "models": [{"id": m["id"], "name": MODEL_NAMES.get(m["id"], m["name"])} for m in saved_metrics],
        "metrics": saved_metrics,
        "dataset": dataset_stats,
    }

@app.get("/analytics")
def get_analytics():
    return {
        "roc_curves":          roc_data,
        "confusion_matrices":  cm_data,
        "feature_importance":  feature_importance,
    }

@app.post("/predict")
def predict(person: PersonInput, model: Optional[str] = None):
    if model and model not in PIPELINES:
        raise HTTPException(status_code=400, detail=f"Modelo desconocido: {model}")

    X = build_feature_vector(person)
    targets = {model: PIPELINES[model]} if model else PIPELINES

    results = []
    for model_id, pipe in targets.items():
        prob = float(pipe.predict_proba(X)[0, 1])
        pred = int(pipe.predict(X)[0])
        results.append({
            "model_id":           model_id,
            "model_name":         MODEL_NAMES[model_id],
            "prediction":         pred,
            "label":              ">50K" if pred == 1 else "≤50K",
            "probability_gt50k":  round(prob, 4),
            "probability_le50k":  round(1 - prob, 4),
        })

    explanation = explain_prediction(X, person)

    return {"results": results, "explanation": explanation}
