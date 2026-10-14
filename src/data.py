import os
from collections import Counter
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
from exceptions import InvalidDatasetStructureError

SEED = 42

CLASS_MAPPING = {
    "buildings" : 0,
    "forest" : 1,
    "glacier" : 2,
    "mountain" : 3,
    "sea" : 4,
    "street" : 5
}

CLASSES = {
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
}

def save_class_mappings(mapping):
    with open("models/class_names.json", 'w') as f:
        json.dump(mapping, f, indent=4)
    return "class mappings saved sucessfuly"

mis = []

def load_dataset(root_or_manifest, seed=SEED, shuffle=False, buffer_size=128, augment=False):
    dirs = set({})
    x = []
    y = []
    
    for folder in os.listdir(root_or_manifest):
        path = os.path.join(root_or_manifest, folder)
        if os.path.isdir(path):
            dirs.add(folder)
    if len( CLASSES - dirs) != 0 or len(dirs - CLASSES) != 0:
        raise InvalidDatasetStructureError("missing or unexpected directories in the dataset")
    for cls in CLASSES:
        CLASS_ID = CLASS_MAPPING[cls]
        imagespath = os.path.join(root_or_manifest, cls)
        for img in os.listdir(imagespath):
            x.append(os.path.join(imagespath, img))
            y.append(CLASS_ID)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=seed, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=0.50, shuffle=True, random_state=seed, stratify=y_temp)
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    if shuffle==True:
        train_ds = train_ds.shuffle(buffer_size=buffer_size, seed=seed)
    
    train_ds = preprocess(train_ds, augment=augment)
    val_ds = preprocess(val_ds)
    test_ds = preprocess(test_ds)
    return train_ds, val_ds, test_ds

def image_process(image_path, image_size=(150, 150)):
    image_encode = tf.io.read_file(image_path)
    image_decode = tf.io.decode_jpeg(image_encode, channels=3)
    resized_image = tf.image.resize(image_decode, image_size)
    return resized_image

def preprocess(ds, batch_size=32, augment=False):
    processed_images = ds.map(lambda x, y: (image_process(x), y))
    if augment == True:
        processed_images = processed_images.map(lambda x, y: (augment_image(x), y))
    batched_images = processed_images.batch(batch_size=batch_size)
    prefetched = batched_images.prefetch(tf.data.AUTOTUNE)
    return prefetched


# augmentation layers, called only once
flip = tf.keras.layers.RandomFlip(mode='horizontal')
rotate = tf.keras.layers.RandomRotation(factor=0.05)
zoom = tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1)
translate = tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05)
contrast = tf.keras.layers.RandomContrast(factor=0.1)

#actual function that reuses the layers
def augment_image(image):
    flipped_image = flip(image)
    rotated_image = rotate(flipped_image)
    zoomed_image = zoom(rotated_image)
    translated_image = translate(zoomed_image)
    contrasted_image = contrast(translated_image)
    return contrasted_image

def count_files_by_type(directory):
    filecounts = Counter()
    totalfiles = 0
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            _, ext = os.path.splitext(filename)
            ext = ext if ext else "(no extension)"
            filecounts[ext] += 1
            totalfiles += 1
            
    return totalfiles, filecounts

def select_random_samples(directory, sample_size=24, seed=SEED):
    random.seed(seed)
    filenames = os.listdir(directory)
    sample = random.sample(list(filenames), sample_size)
    return sample



DATASET_ROOT = "data/splits/seg_train"

train_ds, val_ds, test_ds = load_dataset(
    DATASET_ROOT,
    shuffle=True,
    augment=True
)

# 1. Dataset sizes
print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())
print("Test batches:", tf.data.experimental.cardinality(test_ds).numpy())

# 2. Inspect one training batch
images, labels = next(iter(train_ds))

print("\n--- Training batch ---")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Images dtype:", images.dtype)
print("Labels dtype:", labels.dtype)
print("Pixel minimum:", tf.reduce_min(images).numpy())
print("Pixel maximum:", tf.reduce_max(images).numpy())

# 3. Check class distribution in the original dataset
for class_name, class_id in CLASS_MAPPING.items():
    print(
        f"{class_name} (ID {class_id}):",
        sum(1 for label in labels.numpy() if label == class_id),
        "images in this batch"
    )

# 4. Inspect validation batch
val_images, val_labels = next(iter(val_ds))
print("\n--- Validation batch ---")
print("Images shape:", val_images.shape)
print("Labels shape:", val_labels.shape)

# 5. Inspect test batch
test_images, test_labels = next(iter(test_ds))
print("\n--- Test batch ---")
print("Images shape:", test_images.shape)
print("Labels shape:", test_labels.shape)

# 6. Verify split sizes
print("\n--- Split sizes ---")
print("Train samples:", sum(1 for _ in train_ds.unbatch()))
print("Validation samples:", sum(1 for _ in val_ds.unbatch()))
print("Test samples:", sum(1 for _ in test_ds.unbatch()))