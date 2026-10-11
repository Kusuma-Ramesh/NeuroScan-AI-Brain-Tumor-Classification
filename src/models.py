"""Custom CNN + transfer-learning models. Every model takes 0-1 RGB input; the model-specific
ImageNet preprocessing lives INSIDE the graph (Rescaling/Normalization only -> safe .h5 reload)."""
from tensorflow import keras
from tensorflow.keras import layers as L, applications as A
from .config import IMG, CLASSES

BASES = {"resnet50": A.ResNet50, "mobilenetv2": A.MobileNetV2,
         "inceptionv3": A.InceptionV3, "efficientnetb0": A.EfficientNetB0}
ALL = ["custom_cnn", *BASES]

def custom_cnn(n=len(CLASSES)):
    i = x = keras.Input((*IMG, 3))
    for f in (32, 64, 128, 256):
        x = L.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = L.ReLU()(L.BatchNormalization()(x))
        x = L.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = L.Dropout(0.25)(L.MaxPooling2D()(x))
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(0.5)(L.Dense(256, activation="relu")(x))
    return keras.Model(i, L.Dense(n, activation="softmax")(x), name="custom_cnn"), None

def _prep(name, x):
    if name in ("mobilenetv2", "inceptionv3"):
        return L.Rescaling(2.0, -1.0)(x)                       # -> [-1, 1]
    if name == "efficientnetb0":
        return L.Rescaling(255.0)(x)                           # EfficientNet rescales internally
    x = L.Rescaling(255.0)(x)                                  # resnet50: subtract ImageNet mean
    return L.Normalization(mean=[123.68, 116.779, 103.939], variance=[1.0] * 3)(x)

def transfer(name, n=len(CLASSES)):
    i = keras.Input((*IMG, 3))
    base = BASES[name](include_top=False, weights="imagenet", input_tensor=_prep(name, i))
    base.trainable = False
    x = L.GlobalAveragePooling2D(name="gap")(base.output)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.4)(L.Dense(256, activation="relu")(x))
    return keras.Model(i, L.Dense(n, activation="softmax")(x), name=name), base

def build(name):
    return custom_cnn() if name == "custom_cnn" else transfer(name)
