#  Knee X-ray Osteoarthritis Classification

## Overview

This project classifies knee X-ray images into **5 grades** of Osteoarthritis severity using Deep Learning. Two models are trained and evaluated: a **Custom CNN** built from scratch, and **Xception** via Transfer Learning.

---

##  Project Structure

```
Knee X-ray Classification/
├── New_Dataset/
│   ├── train/          (80% of data)
│   ├── val/            (10% of data)
│   └── test/           (10% of data)
├── model.CNN.keras
├── model.Xception.keras
└── Fixed_KneeXray.ipynb
```

---

##  Target Classes

| Index | Class Name | Description              |
|-------|------------|--------------------------|
| 0     | Normal     | Healthy knee             |
| 1     | Doubtful   | Possible OA signs        |
| 2     | Mild       | Mild OA                  |
| 3     | Moderate   | Moderate OA              |
| 4     | Severe     | Severe OA                |

---

## ⚙️ Libraries Used

```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121, VGG16, ResNet50
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import splitfolders
```

---

##  Data Preparation

### 1. Dataset Splitting

```python
import splitfolders

splitfolders.ratio(
    '/content/drive/MyDrive/Rough_knee Xray/Knee X-ray Images',
    output="New_Dataset",
    seed=1345,
    ratio=(.8, 0.1, 0.1)   # train / val / test
)
```

### 2. Hyperparameters

```python
IMG_HEIGHT = 128
IMG_WIDTH  = 128
IMG_SHAPE  = (IMG_HEIGHT, IMG_WIDTH, 3)
BATCH_SIZE = 32
```

### 3. Loading the Dataset (CNN / General)

```python
normalization_layer = tf.keras.layers.Rescaling(1./255)

def prepare_ds(ds, shuffle=True):
    ds = ds.map(lambda x, y: (normalization_layer(x), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    "New_Dataset/train", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
)
test_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    "New_Dataset/test", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    shuffle=False       # important: keeps order consistent for y_true
)
val_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    "New_Dataset/val", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
)

train_ds = prepare_ds(train_ds_raw, shuffle=True)
test_ds  = prepare_ds(test_ds_raw,  shuffle=False)
val_ds   = prepare_ds(val_ds_raw,   shuffle=True)
```

> ⚠️ **Note:** `shuffle=False` on the test set is critical so that `y_true` and `y_pred` stay aligned.

### 4. Extracting y_true

```python
y_true = np.concatenate([y for x, y in test_ds], axis=0)
```

---

##  Callbacks

```python
def get_callbacks(model_name):
    callbacks = []

    # Save the best model based on val_loss
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'model.{model_name}.keras',
        verbose=1, monitor='val_loss', mode='min', save_best_only=True
    )
    callbacks.append(checkpoint)

    # Reduce learning rate when improvement stalls
    anne = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5,
        verbose=2, min_lr=1e-7, min_delta=1e-5, mode='auto'
    )
    callbacks.append(anne)

    # Stop training early if val_loss does not improve for 10 epochs
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10
    )
    callbacks.append(earlystop)

    return callbacks
```

---

##  Model 1: Custom CNN

### Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

model1 = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=IMG_SHAPE),
    MaxPooling2D(2, 2),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(5, activation='softmax')       # 5 output classes
])

model1.summary()
```

### Training

```python
model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=get_callbacks('CNN')
)
```

### Plotting Training Curves

```python
acc       = history.history['accuracy']
val_acc   = history.history['val_accuracy']
loss      = history.history['loss']
val_loss  = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc,     label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss,     label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()

plt.tight_layout()
plt.show()
```

### Evaluation

```python
model = tf.keras.models.load_model('model.CNN.keras')
results = model.evaluate(test_ds, verbose=1)
print(f"Test Loss: {results[0]:.4f} | Test Accuracy: {results[1]:.4f}")
```

### Classification Report

```python
target_names = ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']

y_pred_raw = model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_pred_raw, axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(target_names)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Confusion Matrix - CNN')
plt.show()
```

---

## Model 2: Xception (Transfer Learning)

> ⚠️ **Important:** Xception has its own preprocessing function that scales pixel values to `[-1, 1]`. Do **not** use `Rescaling(1./255)` with it.

### Data Preparation for Xception

```python
xception_preprocess = tf.keras.applications.xception.preprocess_input

def prepare_ds_xception(ds, shuffle=True):
    ds = ds.map(lambda x, y: (xception_preprocess(x), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds_xcep = prepare_ds_xception(train_ds_raw, shuffle=True)
test_ds_xcep  = prepare_ds_xception(test_ds_raw,  shuffle=False)
val_ds_xcep   = prepare_ds_xception(val_ds_raw,   shuffle=True)

y_true_xcep = np.concatenate([y for x, y in test_ds_xcep], axis=0)
```

### Model Architecture

```python
base_model = tf.keras.applications.Xception(
    weights='imagenet',
    input_shape=IMG_SHAPE,
    include_top=False
)
base_model.trainable = True     # Full fine-tuning

model_xcep = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation='softmax')
])

model_xcep.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Training

```python
history = model_xcep.fit(
    train_ds_xcep,
    validation_data=val_ds_xcep,
    epochs=50,
    callbacks=get_callbacks('Xception')
)
```

### Plotting Training Curves

```python
acc       = history.history['accuracy']
val_acc   = history.history['val_accuracy']
loss      = history.history['loss']
val_loss  = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc,     label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy - Xception')
plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss,     label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss - Xception')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()

plt.tight_layout()
plt.show()
```

### Evaluation

```python
model_xcep = tf.keras.models.load_model('model.Xception.keras')
results = model_xcep.evaluate(test_ds_xcep, verbose=1)
print(f"Test Loss: {results[0]:.4f} | Test Accuracy: {results[1]:.4f}")
```

### Classification Report

```python
y_pred_xcep_raw = model_xcep.predict(test_ds_xcep, verbose=1)
predicted_categories = np.argmax(y_pred_xcep_raw, axis=1)
true_categories = y_true_xcep

print(classification_report(
    true_categories, predicted_categories,
    target_names=target_names, digits=4
))
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(true_categories, predicted_categories)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Xception')
plt.show()
```

### ROC Curve — One-vs-Rest

```python
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize the true labels into one-hot format
y_true_bin = label_binarize(true_categories, classes=np.arange(len(target_names)))
y_score    = y_pred_xcep_raw     # use raw probabilities, NOT argmax
n_classes  = y_true_bin.shape[1]

fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(9, 7))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i],
             label=f'ROC - {target_names[i]} (AUC = {roc_auc[i]:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Xception (One-vs-Rest)')
plt.legend(loc='lower right')
plt.show()
```

---

## 🔍 Inference on a New Image

```python
from google.colab import files
from PIL import Image
import io

# Load the best-performing model
model_xcep = tf.keras.models.load_model('model.Xception.keras')

def preprocess_image_xception(image_data, img_height, img_width):
    img = Image.open(image_data).convert('RGB')
    img = img.resize((img_height, img_width))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)           # add batch dimension
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    return img_array

# Upload and predict
uploaded = files.upload()

for fn in uploaded.keys():
    print(f'Uploaded file: "{fn}"')
    img_array = preprocess_image_xception(
        io.BytesIO(uploaded[fn]), IMG_HEIGHT, IMG_WIDTH
    )
    predictions = model_xcep.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name  = target_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    print(f'Predicted: {predicted_class_name} | Confidence: {confidence:.2%}')
```

---

##  Download the Model

```python
from google.colab import files
files.download('model.Xception.keras')
```

---

## 📌 Key Technical Notes

| Point | Details |
|-------|---------|
| **Preprocessing** | CNN uses `Rescaling(1./255)`; Xception uses its own `preprocess_input` (scales to `[-1, 1]`) |
| **shuffle=False** | Required on test set to keep `y_true` and `y_pred` in the same order |
| **ROC Curve input** | Uses raw probabilities (`y_pred_raw`), not `argmax` predictions |
| **Fine-tuning** | `base_model.trainable = True` enables full fine-tuning of all Xception layers |
| **ModelCheckpoint** | Automatically saves the best model based on `val_loss` |
| **EarlyStopping** | Stops training if `val_loss` does not improve for 10 consecutive epochs |

---

##  References & Tools

- [TensorFlow / Keras Documentation](https://www.tensorflow.org/api_docs)
- [Xception Paper — Chollet 2017](https://arxiv.org/abs/1610.02357)
- [scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
- [split-folders](https://pypi.org/project/split-folders/)
- Dataset: Knee X-ray Images (KL Grade 0–4)
