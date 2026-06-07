import numpy as np
from src.preprocessing import preprocess_denoise
from src.segmentation import segment_lesion
from src.features.extractor import extract_features


def build_features_dataset(dataset, log_every=500):
    """Run the canonical pipeline (denoise -> segment -> extract features) over every
    image. This is the SAME sequence used at inference time, which is essential so
    that the features match the distribution the classifier is trained on."""
    X, Y = [], []
    feature_names = None
    n_samples = len(dataset)
    for i in range(n_samples):
        image, label = dataset[i]
        image = preprocess_denoise(image)
        mask = segment_lesion(image)
        names, features = extract_features(image, mask)
        if feature_names is None:
            feature_names = names
        X.append(features)
        Y.append(label)
        if log_every and (i + 1) % log_every == 0:
            print(f"  features extracted: {i + 1}/{n_samples}")
    return np.array(X, dtype=np.float32), np.array(Y), feature_names


def normalize_with_stats(X, mean, std, eps=1e-8):
    return (X - mean) / (std + eps)


def normalize_dataset(X_train, mean=None, std=None, eps=1e-8):
    if mean is None or std is None:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
    return normalize_with_stats(X_train, mean, std, eps), mean, std
