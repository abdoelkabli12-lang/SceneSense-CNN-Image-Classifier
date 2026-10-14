import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from data import count_files_by_type, select_random_samples
from PIL import Image

folders = [
    "./data/splits/seg_train/buildings",
    "./data/splits/seg_train/forest",
    "./data/splits/seg_train/glacier",
    "./data/splits/seg_train/mountain",
    "./data/splits/seg_train/sea",
    "./data/splits/seg_train/street",
]

fold = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

d = defaultdict(list)

for folder in folders:
    total, types = count_files_by_type(folder)
    d['folder'].append(folder.split('/')[-1])  # e.g. "buildings"
    d['total_files'].append(total)
    print(f"\nFolder: {folder}")
    print(f"Total files: {total}")
    print("File types:", dict(types))


x = d['folder']
y = d['total_files']
plt.bar(x, y)
plt.xlabel("Folder")
plt.ylabel("Total Images")
plt.title("Classes Image Distrebution")
plt.show()

for folder in folders:
    images = []
    samples = select_random_samples(folder)
    for sample in samples:
        images.append(Image.open(os.path.join(folder, sample)))
    n = len(images)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    label = os.path.basename(os.path.normpath(folder))

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(label)
        axes[i].set_xlabel(f"Dimensions: {img.width} x {img.height}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for axis in axes[n:]:
        axis.set_visible(False)

    plt.tight_layout()
    plt.show()
