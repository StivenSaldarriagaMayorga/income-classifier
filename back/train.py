"""
Entrena los 3 modelos del Trabajo2 y guarda los artefactos con joblib.
Ejecutar una vez: python train.py
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, OneHotEncoder, KBinsDiscretizer
)
from sklearn.tree import DecisionTreeClassifier

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Cargar dataset ────────────────────────────────────────────────────────
print("Cargando dataset...")
cols = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
df["income"] = (df["income"].str.strip() == ">50K").astype(int)
df = df.replace("?", np.nan)

# ── 2. Imputación ────────────────────────────────────────────────────────────
cat_cols = ["workclass", "occupation", "native-country"]
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# ── 3. Feature engineering (replica exacta del notebook) ────────────────────
# age_entropia
kbd_age = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
df["age_entropia"] = kbd_age.fit_transform(df[["age"]]).astype(int)

# log transforms
df["capital-gain_log"] = np.log1p(df["capital-gain"])
df["capital-loss_log"] = np.log1p(df["capital-loss"])

# hours-per-week robust
rob = RobustScaler()
df["hours-per-week_robust"] = rob.fit_transform(df[["hours-per-week"]])

# OHE categoricas
ohe_cols = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
ohe = OneHotEncoder(sparse_output=False, dtype=int, drop="first", handle_unknown="ignore")
encoded = ohe.fit_transform(df[ohe_cols])
df_ohe = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)

df_final = pd.concat(
    [df[["age_entropia"]], df[["age", "capital-gain_log", "capital-loss_log"]], df_ohe],
    axis=1,
)
df_final["income"] = df["income"].values
df_final["education-num"] = df["education-num"].values
df_final["hours-per-week"] = df["hours-per-week_robust"].values

# ── 4. Split ─────────────────────────────────────────────────────────────────
X = df_final.drop(columns=["income"])
y = df_final["income"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Entrenar modelos ──────────────────────────────────────────────────────
print("Entrenando Regresión Logística...")
pipe_lr = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=42))])
pipe_lr.fit(X_train, y_train)

print("Entrenando Árbol de Decisión (GridSearchCV)...")
pipe_tree = Pipeline([("sc", StandardScaler()), ("m", DecisionTreeClassifier(random_state=42))])
grid_tree = GridSearchCV(
    pipe_tree,
    {"m__max_depth": [3, 5, 7, 10, None], "m__min_samples_split": [2, 5, 10], "m__criterion": ["gini", "entropy"]},
    cv=5, scoring="roc_auc", n_jobs=-1, refit=True,
)
grid_tree.fit(X_train, y_train)
best_tree = grid_tree.best_estimator_

print("Entrenando KNN (GridSearchCV)...")
pipe_knn = Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier())])
grid_knn = GridSearchCV(
    pipe_knn,
    {"m__n_neighbors": [3, 5, 7, 9, 11, 15, 21]},
    cv=5, scoring="roc_auc", n_jobs=-1, refit=True,
)
grid_knn.fit(X_train, y_train)
best_knn = grid_knn.best_estimator_

# ── 6. Métricas ──────────────────────────────────────────────────────────────
def metrics(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    return {
        "name": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_prob), 4),
    }

model_metrics = [
    metrics("Regresión Logística", pipe_lr, X_test, y_test),
    metrics("Árbol de Decisión", best_tree, X_test, y_test),
    metrics("KNN", best_knn, X_test, y_test),
]
print("\nMétricas en test:")
for m in model_metrics:
    print(f"  {m['name']}: AUC={m['auc']} | Acc={m['accuracy']} | F1={m['f1']}")

# ── 7. Guardar artefactos ────────────────────────────────────────────────────
print("\nGuardando artefactos...")
joblib.dump(pipe_lr,    os.path.join(MODELS_DIR, "logistic_regression.pkl"))
joblib.dump(best_tree,  os.path.join(MODELS_DIR, "decision_tree.pkl"))
joblib.dump(best_knn,   os.path.join(MODELS_DIR, "knn.pkl"))
joblib.dump(kbd_age,    os.path.join(MODELS_DIR, "kbd_age.pkl"))
joblib.dump(rob,        os.path.join(MODELS_DIR, "robust_scaler.pkl"))
joblib.dump(ohe,        os.path.join(MODELS_DIR, "ohe.pkl"))
joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
joblib.dump(model_metrics, os.path.join(MODELS_DIR, "metrics.pkl"))

print("✓ Artefactos guardados en ./models/")
