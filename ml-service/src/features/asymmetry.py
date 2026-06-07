import cv2 as cv
import numpy as np

ASYM_KEYS = ["asymmetry_major", "asymmetry_minor"]


def _zeros():
    return {k: 0.0 for k in ASYM_KEYS}


def extract_asymmetry(mask):
    """ABCD 'A': asymmetry of the lesion shape.

    The mask is aligned to its principal axes (via PCA on the lesion pixels), then
    folded across the major and minor axes. The non-overlapping fraction measures
    asymmetry: 0 = perfectly symmetric, higher = more asymmetric. Melanomas tend to
    be more asymmetric than benign nevi."""
    features = _zeros()
    ys, xs = np.where(mask > 0)
    if xs.size < 10:
        return features

    coords = np.column_stack((xs, ys)).astype(np.float64)
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Largest eigenvalue -> major axis
    major = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(major[1], major[0]))

    h, w = mask.shape
    center = (float(mean[0]), float(mean[1]))
    rot = cv.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv.warpAffine(mask, rot, (w, h), flags=cv.INTER_NEAREST)
    aligned = (aligned > 0).astype(np.uint8)

    area = float(aligned.sum())
    if area == 0:
        return features

    flip_h = cv.flip(aligned, 1)  # across vertical line -> tests major-axis symmetry
    flip_v = cv.flip(aligned, 0)
    diff_major = np.logical_xor(aligned, flip_h).sum()
    diff_minor = np.logical_xor(aligned, flip_v).sum()

    features["asymmetry_major"] = diff_major / (2.0 * area)
    features["asymmetry_minor"] = diff_minor / (2.0 * area)
    return features
