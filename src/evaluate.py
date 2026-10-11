"""Test-set evaluation (accuracy, P/R/F1, confusion matrix, ROC-AUC, latency) + model comparison."""
import json, time
import numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow import keras
from .config import *
from .data import load

def evaluate(name):
    path = MODELS / f"{name}.h5"
    m = keras.models.load_model(path, compile=False)
    ds = load("test")
    y = np.concatenate([b.argmax(1) for _, b in ds.as_numpy_iterator()])
    t0 = time.perf_counter(); p = m.predict(ds, verbose=0)
    lat = (time.perf_counter() - t0) / len(p) * 1000
    rep = classification_report(y, p.argmax(1), target_names=CLASSES, output_dict=True, zero_division=0)
    Y, roc = np.eye(len(CLASSES))[y], {}
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(Y[:, i], p[:, i])
        k = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)
        roc[c] = {"fpr": fpr[k].tolist(), "tpr": tpr[k].tolist(), "auc": float(roc_auc_score(Y[:, i], p[:, i]))}
    out = {"model": name, "accuracy": rep["accuracy"],
           "precision_macro": rep["macro avg"]["precision"], "recall_macro": rep["macro avg"]["recall"],
           "f1_macro": rep["macro avg"]["f1-score"], "auc_macro": float(np.mean([r["auc"] for r in roc.values()])),
           "params": int(m.count_params()), "size_mb": path.stat().st_size / 1e6, "latency_ms": lat,
           "confusion": confusion_matrix(y, p.argmax(1)).tolist(),
           "per_class": {c: rep[c] for c in CLASSES}, "roc": roc}
    json.dump(out, open(REPORTS / f"{name}_metrics.json", "w"))
    return out

def compare():
    """Rank = 70% macro-F1 (reliability) + 15% speed + 15% size (efficiency)."""
    rows = [json.load(open(f)) for f in REPORTS.glob("*_metrics.json")]
    cols = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_macro", "params", "size_mb", "latency_ms"]
    df = pd.DataFrame(rows)[cols]
    n = lambda s: (s.max() - s) / (s.max() - s.min() + 1e-9)
    df["score"] = 0.7 * df.f1_macro + 0.15 * n(df.latency_ms) + 0.15 * n(df.size_mb)
    df = df.sort_values("score", ascending=False)
    df.to_csv(REPORTS / "comparison.csv", index=False)
    json.dump({"best": df.iloc[0].model}, open(REPORTS / "best_model.json", "w"))
    return df

if __name__ == "__main__":      # re-evaluate every saved model
    for f in sorted(MODELS.glob("*.h5")):
        evaluate(f.stem)
    print(compare())
