# Skin Cancer Detection from Dermatoscopic Images  
MobileNetV2 + XGBoost & LightGBM for Multi-Class Classification (ISIC Dataset)

This project builds an end-to-end hybrid deep learning + machine learning pipeline for automated skin cancer detection using dermatoscopic images from the ISIC dataset.  
It performs advanced image preprocessing, deep feature extraction using MobileNetV2, and final multi-class classification using XGBoost and LightGBM, evaluated using clinical-grade metrics like ROC-AUC, F1-score, and Cohen’s Kappa.

---

## The Repository Contains

- A complete training & evaluation notebook  
- Hybrid MobileNetV2 + XGBoost / LightGBM pipeline  
- Advanced image preprocessing & augmentation  
- Model evaluation with:
  - ROC-AUC
  - Confusion matrix
  - Precision, Recall, F1-score
  - Kappa score

---

## Project Overview

Early detection of skin cancer is critical for improving patient survival rates.  
This project trains a hybrid system to classify dermatoscopic images into 9 different skin lesion categories.

The focus is on:

- Robust image preprocessing (hair removal, CLAHE, resizing)
- Deep feature extraction using transfer learning
- Gradient boosting–based classification
- Handling multi-class imbalance
- Clinically relevant evaluation metrics

---

## Dataset 

- Dataset Name: Skin Cancer: 9 Classes – ISIC

- Total Images: 2,357

- Classes: 9 different skin lesion types

- Source: Kaggle (ISIC Archive

---

## Model Architecture

### 1. MobileNetV2 (Feature Extraction)
- ImageNet pretrained  
- Input size: 224 × 224  
- Global Average Pooling  
- Output: 1280-dimensional feature vector  

### 2. XGBoost (Final Classifier)
- Gradient Boosted Decision Trees  
- Optimized for structured deep features  
- Achieves highest accuracy in this project  

### 3. LightGBM (Final Classifier)
- Histogram-based gradient boosting  
- Faster training with competitive accuracy  

Both classifiers are trained on deep feature embeddings extracted from MobileNetV2.

---

## Preprocessing Pipeline

- Image resizing to 224 × 224  
- Hair removal using:
  - Canny Edge Detection  
  - TELEA Inpainting  
- Contrast enhancement using CLAHE  
- Data augmentation:
  - Rotation  
  - Horizontal flip  
  - Brightness & contrast shift  

---

## Training the Models

Main file: skin_cancer_detection.ipynb


Steps include:

- Dataset loading & preprocessing  
- Data augmentation  
- MobileNetV2 feature extraction  
- XGBoost training  
- LightGBM training  
- Best model selection  
- Full metric-based evaluation
  
---

## Evaluation Results

| Model    | Accuracy | F1 (Macro) | Kappa  | Mean AUC |
|----------|----------|------------|--------|----------|
| XGBoost  | 85.48%   | 0.8179     | 0.8284 | 0.9818   |
| LightGBM | 85.08%   | 0.8104     | 0.8237 | 0.9803   |

XGBoost performs slightly better across all major metrics.

---

## Evaluation Metrics Used

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Cohen’s Kappa  
- ROC-AUC  
- Confusion Matrix  

---

## Co-Authors / Teammates

- Ananya Sharma  
- Aditi Agrawal  

Department of Electronics and Communication Engineering  
Thapar Institute of Engineering & Technology, Patiala

---

## Acknowledgements

- Dataset: ISIC Skin Cancer Dataset (Kaggle)  
- Pretrained Network: MobileNetV2 (TensorFlow/Keras)  
- ML Models: XGBoost & LightGBM  

---

## Disclaimer

This project is strictly for educational and research purposes only and must not be used for real-world medical diagnosis without professional validation.

