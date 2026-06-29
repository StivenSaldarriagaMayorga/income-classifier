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

print("Cargando dataset...")
cols = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

_local_candidates = [
    os.path.join(os.path.dirname(__file__), "adult.data"),
    os.path.join(os.path.dirname(__file__), "adult.csv"),
    os.path.join(os.path.dirname(__file__), "adult.data.txt"),
]
_local_file = next((p for p in _local_candidates if os.path.exists(p)), None)

if _local_file:
    print(f"  Usando archivo local: {_local_file}")
    df = pd.read_csv(_local_file, names=cols, na_values=" ?", skipinitialspace=True)
else:
    print("  Intentando descarga remota...")
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
    except Exception:
        print("  Descarga fallida. Intentando sklearn fetch_openml...")
        from sklearn.datasets import fetch_openml
        raw = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df = raw.frame.copy()
        col_map = {
            "age": "age", "workclass": "workclass", "fnlwgt": "fnlwgt",
            "education": "education", "education-num": "education-num",
            "marital-status": "marital-status", "occupation": "occupation",
            "relationship": "relationship", "race": "race", "sex": "sex",
            "capital-gain": "capital-gain", "capital-loss": "capital-loss",
            "hours-per-week": "hours-per-week", "native-country": "native-country",
            "class": "income",
        }
        df = df.rename(columns=col_map)
        df["income"] = df["income"].astype(str).str.strip()

df["income"] = df["income"].astype(str).str.strip()
df["income"] = df["income"].map(lambda x: 1 if x in (">50K", ">50K.") else 0)
df = df.replace("?", np.nan)

cat_cols = ["workclass", "occupation", "native-country"]
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

kbd_age = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
df["age_entropia"] = kbd_age.fit_transform(df[["age"]]).astype(int)

df["capital-gain_log"] = np.log1p(df["capital-gain"])
df["capital-loss_log"] = np.log1p(df["capital-loss"])

rob = RobustScaler()
df["hours-per-week_robust"] = rob.fit_transform(df[["hours-per-week"]])

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

X = df_final.drop(columns=["income"])
y = df_final["income"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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

from sklearn.metrics import roc_curve, confusion_matrix, precision_score, recall_score

MODELS_LIST = [
    ("logistic_regression", "Regresion Logistica", pipe_lr),
    ("decision_tree",       "Arbol de Decision",   best_tree),
    ("knn",                 "KNN",                 best_knn),
]

model_metrics = []
roc_data      = []
cm_data       = []

for model_id, name, pipe in MODELS_LIST:
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc  = round(accuracy_score(y_test, y_pred), 4)
    f1   = round(f1_score(y_test, y_pred), 4)
    auc  = round(roc_auc_score(y_test, y_prob), 4)
    prec = round(precision_score(y_test, y_pred), 4)
    rec  = round(recall_score(y_test, y_pred), 4)

    model_metrics.append({
        "id": model_id, "name": name,
        "accuracy": acc, "f1": f1, "auc": auc,
        "precision": prec, "recall": rec,
    })

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
    roc_data.append({
        "id": model_id, "name": name, "auc": auc,
        "fpr": fpr[idx].round(4).tolist(),
        "tpr": tpr[idx].round(4).tolist(),
    })

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_data.append({
        "id": model_id, "name": name,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    })

    print(f"  {name}: AUC={auc} | Acc={acc} | F1={f1} | Prec={prec} | Rec={rec}")

lr_coefs = np.abs(pipe_lr.named_steps["m"].coef_[0])
feat_imp = sorted(
    zip(feature_names, lr_coefs.round(4).tolist()),
    key=lambda x: x[1], reverse=True
)[:20]
feature_importance = [{"feature": f, "importance": v} for f, v in feat_imp]

dataset_stats = {
    "total_samples": len(X),
    "train_samples": len(X_train),
    "test_samples":  len(X_test),
    "n_features":    len(feature_names),
    "class_balance": {
        "le50k": int((y == 0).sum()),
        "gt50k": int((y == 1).sum()),
    }
}

print("\nGuardando artefactos...")
joblib.dump(pipe_lr,           os.path.join(MODELS_DIR, "logistic_regression.pkl"))
joblib.dump(best_tree,         os.path.join(MODELS_DIR, "decision_tree.pkl"))
joblib.dump(best_knn,          os.path.join(MODELS_DIR, "knn.pkl"))
joblib.dump(kbd_age,           os.path.join(MODELS_DIR, "kbd_age.pkl"))
joblib.dump(rob,               os.path.join(MODELS_DIR, "robust_scaler.pkl"))
joblib.dump(ohe,               os.path.join(MODELS_DIR, "ohe.pkl"))
joblib.dump(feature_names,     os.path.join(MODELS_DIR, "feature_names.pkl"))
joblib.dump(model_metrics,     os.path.join(MODELS_DIR, "metrics.pkl"))
joblib.dump(roc_data,          os.path.join(MODELS_DIR, "roc_data.pkl"))
joblib.dump(cm_data,           os.path.join(MODELS_DIR, "cm_data.pkl"))
joblib.dump(feature_importance,os.path.join(MODELS_DIR, "feature_importance.pkl"))
joblib.dump(dataset_stats,     os.path.join(MODELS_DIR, "dataset_stats.pkl"))

print("OK: Artefactos guardados en ./models/")
