import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf

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

def select_random_samples(directory, sample_size=24):
    random.seed(42)
    filenames = os.listdir(directory)
    sample = random.sample(list(filenames), sample_size)
    return sample

def split_dataset(directory):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(224, 224),
        batch_size=32,
        seed=42
    )
    
    total = int(dataset.cardinality().numpy())
    total_samples = total * 32
    
    train_size = int(0.70 * total)
    val_size   = int(0.15 * total)
    
    train_ds = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.take(val_size)
    return train_ds, val_ds, test_ds, total_samples
