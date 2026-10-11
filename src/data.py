"""Data loading, preprocessing (0-1 scaling, 224x224), augmentation and dataset stats."""
import json
from collections import Counter
import tensorflow as tf
from PIL import Image
from .config import *

AUG = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.2, value_range=(0, 1)),
])

def load(split, train=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA / split, image_size=IMG, batch_size=BATCH, label_mode="categorical",
        class_names=CLASSES, shuffle=train, seed=SEED)
    ds = ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.map(lambda x, y: (AUG(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def class_weights():
    counts = [len(list((DATA / "train" / c).glob("*"))) for c in CLASSES]
    return {i: sum(counts) / (len(CLASSES) * n) for i, n in enumerate(counts)}

def dataset_stats():
    """Class balance per split + resolution consistency -> reports/dataset_stats.json"""
    counts = {s: {c: len(list((DATA / s / c).glob("*"))) for c in CLASSES} for s in ("train", "valid", "test")}
    sizes = Counter(Image.open(p).size for c in CLASSES for p in list((DATA / "train" / c).glob("*"))[:100])
    out = {"counts": counts, "sizes": {f"{w}x{h}": n for (w, h), n in sizes.most_common(5)}}
    REPORTS.mkdir(exist_ok=True)
    json.dump(out, open(REPORTS / "dataset_stats.json", "w"), indent=1)
    return out

if __name__ == "__main__":
    print(dataset_stats())
