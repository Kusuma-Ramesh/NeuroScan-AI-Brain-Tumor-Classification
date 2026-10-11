"""Train -> evaluate -> compare.   python -m src.train --models custom_cnn mobilenetv2 --epochs 30"""
import argparse, json
from tensorflow import keras
from .config import *
from .data import load, class_weights, dataset_stats
from .models import build, ALL
from .evaluate import evaluate, compare

def callbacks(name):
    return [keras.callbacks.EarlyStopping("val_loss", patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=3),
            keras.callbacks.ModelCheckpoint(str(MODELS / f"{name}.h5"), monitor="val_loss", save_best_only=True)]

def compile_(m, lr):
    m.compile(keras.optimizers.Adam(lr), "categorical_crossentropy", metrics=["accuracy"])

def fit(name, epochs, ft_epochs):
    train, val = load("train", True), load("valid")
    m, base = build(name); hist = {}
    def run(n):
        h = m.fit(train, validation_data=val, epochs=n, class_weight=class_weights(), callbacks=callbacks(name))
        for k, v in h.history.items():
            hist.setdefault(k, []).extend(float(x) for x in v)
    compile_(m, 1e-3); run(epochs)
    if base is not None and ft_epochs:                    # fine-tune top 30 layers (BatchNorm stays frozen)
        base.trainable = True
        for l in base.layers[:-30]: l.trainable = False
        for l in base.layers:
            if isinstance(l, keras.layers.BatchNormalization): l.trainable = False
        compile_(m, 1e-5); run(ft_epochs)
    m.save(MODELS / f"{name}.h5")
    json.dump(hist, open(REPORTS / f"{name}_history.json", "w"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ft-epochs", type=int, default=10)
    a = ap.parse_args()
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    dataset_stats()
    for n in a.models:
        fit(n, a.epochs, a.ft_epochs); print(n, "->", evaluate(n)["accuracy"])
    print(compare())
