# SceneSense-CNN-Image-Classifier
A TensorFlow/Keras CNN that classifies images into six scene categories, with data preprocessing, training-only augmentation, model evaluation, error analysis, and single-image prediction.

## Visualize preprocessing and augmentation

Run `python src/visualisation.py` from the repository root to display the dataset plots and a three-panel comparison of an example image: original, resized/preprocessed, and augmented.

To compare a specific image, call `plot_augmentation_comparison(image_path)` from `src/visualisation.py`.
