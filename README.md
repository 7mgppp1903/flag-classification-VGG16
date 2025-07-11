# South American Flag Classification with VGG16 + XGBoost

![Model Visualization](https://github.com/7mgppp1903/flag-classification-vgg16/blob/main/Results/visualisation.png)

## Project Overview

This project performs **highly accurate image classification** of South American country flags using a **hybrid deep learning approach**. I utilize **VGG16 for feature extraction** and **XGBoost for classification**, achieving **97% test accuracy** on a custom dataset of over **10,000 flag images**.

---

##  Key Features

- Hybrid architecture combining VGG16 with XGBoost
- 97% test accuracy on South American flags  
- Comprehensive training and evaluation pipeline   
- Detailed performance visualizations  

---

## Model Architecture

```mermaid
graph LR
A[Input Image] --> B[VGG16 Feature Extraction]
B --> C[Remove Final Layers]
C --> D[XGBoost Classifier]
D --> E[Country Prediction]

```
---
## 📈 Performance Results

### 🔢 Accuracy Metrics

| Metric         | Value               |
|----------------|---------------------|
| Accuracy       | **97.0%**           |
| Dataset Size   | ~10,000 images      |
| Classes        | 12 South American countries |

---

### Confusion Matrix

![Confusion Matrix](https://github.com/7mgppp1903/flag-classification-vgg16/blob/main/Results/confusion%20matrix.png)

### Training Progress

| Accuracy Progress | Loss Reduction |
|------------------|----------------|
| ![Accuracy](https://github.com/7mgppp1903/flag-classification-vgg16/blob/main/Results/model%20accuracy.png) | ![Loss](https://github.com/7mgppp1903/flag-classification-vgg16/blob/main/Results/model%20loss.png) |

---




