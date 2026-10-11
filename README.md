# NeuroScan AI — Brain Tumor MRI Classification

A deep-learning project that classifies brain MRI images into four dataset categories using a custom Convolutional Neural Network (CNN) and four ImageNet-pretrained transfer-learning models. It includes evaluation reports and an interactive Streamlit dashboard with Grad-CAM visual explanations.

> **Medical disclaimer:** This project is for educational and research purposes only. It is not a medical device and must not be used to diagnose, treat, or make clinical decisions.

## 1. Project Overview

NeuroScan AI explores brain MRI image classification for four categories:

- Glioma
- Meningioma
- No tumor
- Pituitary

The application allows users to upload an MRI image, select a model, view predicted class probabilities, and inspect a Grad-CAM heatmap. Predictions are experimental and are not a substitute for assessment by a qualified medical professional.

## 2. Features

- Dataset exploration and class-distribution statistics
- Image resizing to 224 × 224 pixels and pixel scaling to the 0–1 range
- Data augmentation using flips, rotation, zoom, translation, and brightness changes
- Custom CNN with convolutional layers, batch normalization, and dropout
- Transfer learning with ResNet50, MobileNetV2, InceptionV3, and EfficientNetB0
- Class-weighted training and training callbacks
- Evaluation using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, model size, and inference latency
- Grad-CAM visual explanations
- Streamlit dashboard with Overview, Diagnose, Performance, Comparison, and Dataset sections
- Saved model files and evaluation reports

## 3. Repository Structure

```text
NeuroScan-AI-Brain-Tumor-Classification/
├── app/
│   └── app.py
├── data/
│   └── .gitkeep
├── models/
│   ├── custom_cnn.h5
│   ├── efficientnetb0.h5
│   ├── inceptionv3.h5
│   ├── mobilenetv2.h5
│   └── resnet50.h5
├── notebooks/
│   └── Brain_Tumor_MRI_Classification.ipynb
├── reports/
│   ├── comparison.csv
│   ├── dataset_stats.json
│   ├── best_model.json
│   └── model history and metrics JSON files
├── src/
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── models.py
│   └── train.py
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```

## 4. Dataset

**Source:** [Brain Tumor MRI Multi-Class Dataset — Google Drive](https://drive.google.com/drive/folders/1C9ww4JnZ2sh22I-hbt45OR16o4ljGxju?usp=sharing)

The dataset is not included in this GitHub repository. Download it from the source and extract it into the following structure:

```text
data/
└── Tumour/
    ├── train/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── no_tumor/
    │   └── pituitary/
    ├── valid/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── no_tumor/
    │   └── pituitary/
    └── test/
        ├── glioma/
        ├── meningioma/
        ├── no_tumor/
        └── pituitary/
```

The current dataset contains **2,443 images**:

| Split | Images |
|---|---:|
| Training | 1,695 |
| Validation | 502 |
| Testing | 246 |
| **Total** | **2,443** |

Class counts and dataset statistics are available in `reports/dataset_stats.json`.

## 5. Installation and Setup

### Prerequisites

- Python 3.11 recommended
- Git
- Git LFS for downloading the saved model files
- The dataset downloaded separately from the source above

### Step 1: Clone the repository

```powershell
git lfs install
git clone https://github.com/Kusuma-Ramesh/NeuroScan-AI-Brain-Tumor-Classification.git
cd NeuroScan-AI-Brain-Tumor-Classification
```

### Step 2: Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Add the dataset

Download and extract the dataset into `data/Tumour/`. Ensure that the `train`, `valid`, and `test` directories each contain the four class folders shown above.

### Step 5: Generate dataset statistics

```powershell
python -m src.data
```

### Step 6: Launch the application

```powershell
streamlit run app/app.py
```

Open the local URL printed in the terminal by Streamlit.

## 6. Model Training

The training script supports the custom CNN and four pretrained architectures.

To train all five models using the short CPU configuration used for the transfer-learning comparison:

```powershell
python -m src.train --models custom_cnn mobilenetv2 efficientnetb0 resnet50 inceptionv3 --epochs 5 --ft-epochs 0
```

**Note:** This command retrains the models and can take considerable time on a CPU. It is not necessary to run it just to use the existing saved models.

The reported transfer-learning models were trained for five epochs with their pretrained backbones frozen. Fine-tuning was not performed for these runs. The custom CNN received only a two-epoch smoke test and did not achieve competitive performance.

To run that short custom CNN smoke test again:

```powershell
python -m src.train --models custom_cnn --epochs 2 --ft-epochs 0
```

## 7. Model Comparison

The following results are from the current evaluation report, `reports/comparison.csv`. Metrics were calculated using the project's test split.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro ROC-AUC | Size (MB) | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MobileNetV2** | **84.15%** | 84.89% | 84.90% | **84.33%** | 97.14% | 13.58 | 29.39 |
| ResNet50 | 83.33% | 84.36% | 84.17% | 83.58% | **97.41%** | 101.33 | 173.95 |
| InceptionV3 | 80.49% | 80.49% | 80.64% | 79.66% | 96.06% | 94.59 | 83.75 |
| EfficientNetB0 | 79.67% | 80.24% | 80.12% | 79.02% | 95.99% | 20.94 | 87.16 |
| Custom CNN | 32.52% | 8.13% | 25.00% | 12.27% | 75.19% | 15.00 | 40.79 |

### Results interpretation

**MobileNetV2 is the strongest practical candidate among these saved runs.** It achieved the highest accuracy and macro F1-score while remaining relatively small and fast. ResNet50 achieved a slightly higher ROC-AUC but has substantially greater model size and inference latency.

The custom CNN performed poorly in its short smoke test. Its reported result is a baseline rather than evidence of a fully trained, competitive model.

Latency depends on the hardware and evaluation environment and should not be interpreted as a universal benchmark. These results do not establish clinical safety or diagnostic reliability.

## 8. Project Components

| Component | Purpose |
|---|---|
| `src/data.py` | Dataset loading, resizing, scaling, augmentation, class weights, and statistics |
| `src/models.py` | Custom CNN and four pretrained model architectures |
| `src/train.py` | Model training, callbacks, and optional fine-tuning |
| `src/evaluate.py` | Classification metrics, confusion matrices, ROC-AUC, latency, and comparison |
| `src/gradcam.py` | Grad-CAM heatmap generation |
| `app/app.py` | Streamlit application |
| `notebooks/Brain_Tumor_MRI_Classification.ipynb` | Project notebook |
| `reports/` | Dataset statistics, model metrics, training histories, and comparison results |
| `models/` | Saved `.h5` models tracked with Git LFS |

## 9. Limitations and Responsible Use

- This is an educational prototype, not a clinically validated system.
- Transfer-learning results come from five frozen-backbone epochs without fine-tuning.
- The custom CNN received only a short smoke test and performed poorly.
- Results depend on the dataset, labels, preprocessing, train-validation-test split, and computing environment.
- The dataset should be checked for duplicate or near-duplicate images across splits before making stronger generalization claims.
- Grad-CAM provides a visual inspection aid; it does not prove that a prediction is correct or that the model uses clinically meaningful features.
- The application must not be used for real patient care, diagnosis, or treatment decisions.

## 10. Skills Demonstrated

Deep Learning · Python · TensorFlow/Keras · Image Preprocessing · Data Augmentation · Convolutional Neural Networks · Transfer Learning · Computer Vision · Model Evaluation · Grad-CAM · Streamlit Deployment

## License

No license has been specified. Unless a license is added, do not assume that this repository is available for unrestricted reuse.
