import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data import (
    augment_image,
    count_files_by_type,
    image_process,
    select_random_samples,
)


FOLDERS = [
    "./data/splits/seg_train/buildings",
    "./data/splits/seg_train/forest",
    "./data/splits/seg_train/glacier",
    "./data/splits/seg_train/mountain",
    "./data/splits/seg_train/sea",
    "./data/splits/seg_train/street",
]


def plot_augmentation_comparison(image_path, image_size=(150, 150)):
    """Plot an image before preprocessing and after the training augmentation."""
    with Image.open(image_path) as image:
        original_image = image.convert("RGB")

    preprocessed_image = image_process(image_path, image_size=image_size)
    augmented_image = augment_image(preprocessed_image)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[1].imshow(np.clip(preprocessed_image.numpy(), 0, 255).astype(np.uint8))
    axes[1].set_title(f"Preprocessed ({image_size[0]} × {image_size[1]})")
    axes[2].imshow(np.clip(augmented_image.numpy(), 0, 255).astype(np.uint8))
    axes[2].set_title("After augmentation")

    for axis in axes:
        axis.set_axis_off()

    figure.suptitle("Image preprocessing and data augmentation")
    figure.tight_layout()
    plt.show()
    return figure, axes


def plot_dataset_overview():
    image_counts = defaultdict(list)

    for folder in FOLDERS:
        total, types = count_files_by_type(folder)
        image_counts["folder"].append(os.path.basename(folder))
        image_counts["total_files"].append(total)
        print(f"\nFolder: {folder}")
        print(f"Total files: {total}")
        print("File types:", dict(types))

    plt.bar(image_counts["folder"], image_counts["total_files"])
    plt.xlabel("Folder")
    plt.ylabel("Total Images")
    plt.title("Class Image Distribution")
    plt.show()

    for folder in FOLDERS:
        images = []
        samples = select_random_samples(folder)
        for sample in samples:
            images.append(Image.open(os.path.join(folder, sample)))
        count = len(images)
        columns = 4
        rows = (count + columns - 1) // columns

        figure, axes = plt.subplots(rows, columns, figsize=(15, 10))
        axes = axes.flatten()
        label = os.path.basename(os.path.normpath(folder))

        for index, image in enumerate(images):
            axes[index].imshow(image)
            axes[index].set_title(label)
            axes[index].set_xlabel(f"Dimensions: {image.width} x {image.height}")
            axes[index].set_xticks([])
            axes[index].set_yticks([])

        for axis in axes[count:]:
            axis.set_visible(False)

        figure.tight_layout()
        plt.show()


def main():
    plot_dataset_overview()
    sample_path = os.path.join(
        FOLDERS[0],
        select_random_samples(FOLDERS[0], sample_size=1)[0],
    )
    plot_augmentation_comparison(sample_path)


if __name__ == "__main__":
    main()
