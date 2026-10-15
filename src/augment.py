
import time
import tensorflow as tf
from data import load_dataset

# Use the same pipeline settings as your training run
train_ds, _, _ = load_dataset(
    "data/splits/seg_train",
    shuffle=True,
    augment=True,
    buffer_size=1000
)

num_batches = 100
batch_times = []

iterator = iter(train_ds)

for i in range(num_batches):
    start = time.perf_counter()

    try:
        images, labels = next(iterator)

        # Force TensorFlow to finish producing this batch
        images.numpy()
        labels.numpy()

    except StopIteration:
        break

    batch_times.append(time.perf_counter() - start)

if batch_times:
    total_time = sum(batch_times)

    print(f"Batches processed: {len(batch_times)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per batch: {total_time / len(batch_times):.4f} seconds")
    print(f"Images processed: {sum(int(x.shape[0]) for x in [])}")
else:
    print("No batches were processed.")
