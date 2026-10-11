"""Grad-CAM: highlights the MRI regions that drove the prediction (explainability)."""
import numpy as np, tensorflow as tf

def gradcam(model, x, idx=None):
    gm = tf.keras.Model(model.inputs, [model.get_layer("gap").input, model.output])
    with tf.GradientTape() as tape:
        conv, p = gm(x)
        idx = int(tf.argmax(p[0])) if idx is None else idx
        score = p[:, idx]
    w = tf.reduce_mean(tape.gradient(score, conv), axis=(1, 2))
    h = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], -1))[0].numpy()
    return h / (h.max() + 1e-8)
