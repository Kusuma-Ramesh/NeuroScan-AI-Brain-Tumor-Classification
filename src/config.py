"""Central configuration (no TensorFlow import so the dashboard stays light)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Tumour"          # contains train/ valid/ test/
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]  # alphabetical = Keras order
IMG, BATCH, SEED = (224, 224), 32, 42
