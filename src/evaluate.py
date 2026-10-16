import json
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
from data import load_dataset
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

DATASET_ROOT = os.path.expanduser("data/splits/seg_train")

_, _, test_ds = load_dataset(DATASET_ROOT)


with open ("models/training_history.json", "r") as f:
  metrics = json.load(f)

epochs = range(1,len(metrics["accuracy"]) + 1)

plt.figure(figsize=(12,12))
plt.title("Accuracy VS Val_accuracy")
plt.plot(epochs, metrics['accuracy'], label="accuracy")
plt.plot(epochs, metrics['val_accuracy'], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(12,12))
plt.title("Loss VS Val_loss")
plt.plot(epochs, metrics['loss'], label="loss")
plt.plot(epochs, metrics['val_loss'], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


plt.figure(figsize=(12,12))
plt.title("Learning Rate")
plt.plot(epochs, metrics['learning_rate'], label="learning_rate")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.legend()
plt.show()

model = tf.keras.models.load_model("models/best_model.keras", compile=True)
model.evaluate(test_ds)



prediction = model.predict(test_ds)

predicted_classes = np.argmax(prediction, axis=1)

true_classes = np.concatenate([
    labels.numpy() for images, labels in test_ds
])

print(true_classes.shape)
print(true_classes[:10])


cm = confusion_matrix(true_classes, predicted_classes)



class_names = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


print("Accuracy:", accuracy_score(true_classes, predicted_classes))
print("Precision (macro):", precision_score(true_classes, predicted_classes, average="macro"))
print("Recall (macro):", recall_score(true_classes, predicted_classes, average='macro'))
print("F1_Score (macro):", f1_score(true_classes, predicted_classes, average='macro'))
print("Precision (weighted):", precision_score(true_classes, predicted_classes, average="weighted"))
print("Recall (weighted):", recall_score(true_classes, predicted_classes, average='weighted'))
print("F1_Score (weighted):", f1_score(true_classes, predicted_classes, average='weighted'))
print("classification_report:", classification_report(true_classes, predicted_classes, target_names=class_names, zero_division=0))