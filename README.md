# NeuroScan AI - Brain Tumor MRI Classification
Custom CNN vs transfer learning (ResNet50, MobileNetV2, InceptionV3, EfficientNetB0) on 4 classes
(glioma, meningioma, no_tumor, pituitary) with a premium Streamlit dashboard and Grad-CAM explainability.

## Quick start
```bash
pip install -r requirements.txt
unzip Tumour-*.zip -d data/            # -> data/Tumour/{train,valid,test}/<class>/
python -m src.train --models custom_cnn mobilenetv2 efficientnetb0 resnet50 inceptionv3 --epochs 5 --ft-epochs 0
streamlit run app/app.py
```
Quick smoke test: `python -m src.train --models custom_cnn --epochs 2 --ft-epochs 0`.
The reported comparison models were trained for 5 frozen-base epochs on CPU. The training script also supports optional fine-tuning.

## Structure
| Path | Purpose |
|---|---|
| `src/data.py` | loading, 0â€“1 scaling, 224Ã—224 resize, augmentation, class weights, dataset stats |
| `src/models.py` | custom CNN (BatchNorm + Dropout) and 4 pretrained backbones |
| `src/train.py` | 2-phase training (frozen head â†’ fine-tune), EarlyStopping / ModelCheckpoint / ReduceLROnPlateau |
| `src/evaluate.py` | accuracy, precision, recall, F1, confusion matrix, ROC-AUC, latency; model ranking |
| `src/gradcam.py` | Grad-CAM heatmaps |
| `app/app.py` | dashboard: Overview Â· Diagnose Â· Performance Â· Comparison Â· Dataset |
| `notebooks/` | `.ipynb` for submission |
| `models/`, `reports/` | generated `.h5` models, metrics JSON, `comparison.csv` |

## Notes
- All numbers in the dashboard come from **your** training run; nothing is hard-coded.
- Best model = 70% macro-F1 + 15% speed + 15% size (edit in `evaluate.compare`).
- Models are tested on the untouched `test` split. Check for near-duplicate images across splits (the file names suggest Roboflow exports).
- Research/education only â€“ not a medical device.

