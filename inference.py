import os
import requests
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from visualize_predictions import visualize_predictions
from confusion_matrix import plot_confusion_matrix

#Download model if not exists
model_url = "https://huggingface.co/7mgppp/flag-classifier-vgg16-model/resolve/main/flag_classifier_vgg16.h5"
model_path = "flag_classifier_vgg16.h5"

if not os.path.exists(model_path):
    print("Downloading model...")
    response = requests.get(model_url)
    response.raise_for_status()
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Model downloaded.")

#Load model
print("Loading model...")
model = load_model(model_path)
print("Model loaded.")

#Load images
def load_images(image_dir, target_size=(128, 128)):
    images, labels = [], []
    for label in os.listdir(image_dir):
        class_dir = os.path.join(image_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = image.load_img(img_path, target_size=target_size)
                img_array = image.img_to_array(img)
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Skipped: {img_path} ({e})")
    return np.array(images), np.array(labels)

dataset_path = "/Users/miilee/Desktop/ML proj/Dataset"  # Update if needed
print("Loading and preprocessing images...")
X, y = load_images(dataset_path)
X = X / 255.0

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_labels = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
print(f"Loaded dataset: {len(X_train)} train, {len(X_test)} test")

#Load training history
history_path = "training_history.pkl"
history = None
if os.path.exists(history_path):
    with open(history_path, "rb") as f:
        history = pickle.load(f)
    print("Training history loaded.")
else:
    print(f"Warning: {history_path} not found. Skipping history plots.")

#Predict
print("Running predictions...")
y_pred = np.argmax(model.predict(X_test), axis=1)

#Visualize predictions
plt.figure(figsize=(14, 8))
visualize_predictions(X_test, y_test, y_pred, class_labels)
plt.show()

plt.figure(figsize=(10, 7))
plot_confusion_matrix(y_test, y_pred, class_labels)
plt.show()

#Plot Accuracy & Loss if history available
if history:
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
