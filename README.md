# 🌪️ Tornado Detection using Pretrained CNN Models

This project implements a deep learning pipeline for **binary image classification** to detect tornadoes in images. It leverages **pretrained convolutional neural networks (CNNs)** and evaluates their performance across multiple datasets, including a stress test set for robustness analysis.

---

##  Overview

The objective of this project is to build and evaluate a reliable model that can:

- Classify images as **tornado (1)** or **non-tornado (0)**
- Compare the performance of multiple pretrained CNN architectures
- Handle **class imbalance** effectively
- Provide detailed evaluation using multiple performance metrics
- Analyze model weaknesses through **false positive analysis**

---

## Models Used

The framework supports several pretrained models from `torchvision`, including:

- ResNet (ResNet50, ResNet101)
- DenseNet121
- EfficientNet (B0, B3)
- MobileNetV3
- ConvNeXt
- VGG16

---

## Dataset Structure

The dataset is organized using CSV files located in the `splits_final/` directory:
splits_final/
├── train.csv
├── val.csv
├── test.csv
└── stress_test.csv


Each CSV file contains:
- Image file paths
- Binary labels (`0 = non-tornado`, `1 = tornado`)
- Optional subclass labels for deeper analysis

---

## Training Configuration

Key training parameters:

- **Image size:** 224 × 224
- **Batch size:** 32
- **Epochs:** 100
- **Loss function:** Binary Cross Entropy with Logits
- **Optimizer:** Adam
- **Class imbalance handling:** `pos_weight`
- **Data augmentation:**
  - Random horizontal flip
  - Color jitter
  - Normalization

---

## Pipeline

The workflow follows these steps:

1. Load datasets from CSV files  
2. Apply preprocessing and data augmentation  
3. Initialize a pretrained CNN model  
4. Replace the final classification layer for binary output  
5. Train the model  
6. Evaluate performance on:
   - Validation set
   - Test set
   - Stress test set  
7. Save predictions, metrics, and logs  

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

Additional analysis includes:

- Training and validation loss curves  
- False positive breakdown by subclass  

---

##  Key Features

### Class Imbalance Handling

### False Positive Analysis

Identifies which non-tornado image categories are most frequently misclassified.

### Stress Testing

Evaluates model robustness using a separate stress test dataset.

### Results
All model results are stored in:

model_outputs_updated/all_model_results.csv

Models are compared based on:

Test F1 Score
ROC-AUC

### How to Run:

#### 1. Install Dependencies
pip install torch torchvision pandas numpy matplotlib pillow opencv-python scikit-learn

#### 2. Update Dataset Paths (if needed)
TRAIN_CSV = "splits_final/train.csv"
VAL_CSV = "splits_final/val.csv"
TEST_CSV = "splits_final/test.csv"

#### 3. Run the Notebook
jupyter notebook pretrained_models_updated.ipynb

