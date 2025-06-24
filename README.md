# Flag Classification of South American Countries 🌎🚩

This project performs **image classification of South American country flags** using a **transfer learning pipeline**. A custom dataset of over **10,000 flag images** was collected and used for training a hybrid deep learning + machine learning model.

---

## 🧠 Model Architecture

- **Base Model**: VGG16 pretrained on ImageNet
- **Modification**: Final classification layer removed
- **Final Classifier**: XGBoost model trained on extracted CNN features

---

## 📊 Results

| Metric     | Value  |
|------------|--------|
| Accuracy   | 97.0%  |
| Dataset    | ~10,000 self-collected flag images |
| Classes    | South American countries (e.g., Brazil, Argentina, Chile, etc.) |

---

## ⚙️ How to Run It

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract features using VGG16

```bash
python train_vgg16.py
```
### 3. Run inference on a new image

```bash
python inference.py --image path/to/flag.jpg
```


