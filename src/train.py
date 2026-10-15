import tensorflow as tf
from data import load_dataset
from model import build_model
import os
import json

DATASET_ROOT = os.path.expanduser("~/data/splits/seg_train")

train_ds, val_ds, test_ds = load_dataset(DATASET_ROOT, augment=True, shuffle=True)

model = build_model()

early_stopping = tf.keras.callbacks.EarlyStopping(
  monitor="val_loss",
  patience=4,
  restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
  filepath="models/best_model.keras",
  monitor="val_loss",
  save_best_only=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
  monitor="val_loss",
  factor=0.5,
  patience=2,
  mode="auto",
  min_delta=0.0001,
  cooldown=0,
  min_lr=1e-6
)

callbacks = [early_stopping, reduce_lr, model_checkpoint]

history = model.fit(
  x=train_ds,
  epochs=36,
  validation_data=val_ds,
  callbacks=callbacks
)

with open ("models/training_history.json", "w") as f:
  json.dump(history.history, f, indent=4)