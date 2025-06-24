import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image


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
            except:
                print(f"⚠️ Skipped: {img_path}")
    return np.array(images), np.array(labels)

#Load dataset
X, y = load_images("/Users/miilee/Desktop/ML proj/Dataset")  # Update if needed
X = X / 255.0

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_labels = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

print("✅ Data loaded and split:")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

#Build model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers[:-4]:
    layer.trainable = False  # Freeze early layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(class_labels), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Split validation data
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)

#Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_tr)

train_gen = datagen.flow(X_tr, y_tr, batch_size=32)
val_gen = datagen.flow(X_val, y_val, batch_size=32)

#Train model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    callbacks=[early_stop],
    verbose=1
)

import pickle

with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("Training history saved to training_history.pkl")


#Save model
model.save("flag_classifier_vgg16.h5")
print(" Model saved!")

#Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f" Test Accuracy: {acc:.2f}")

