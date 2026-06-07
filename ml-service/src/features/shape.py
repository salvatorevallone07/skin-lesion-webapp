import cv2 as cv
import numpy as np

SHAPE_KEYS = ["area_ratio", "circularity", "eccentricity", "solidity",
              "border_irregularity", "extent"]


def _zeros():
    return {k: 0.0 for k in SHAPE_KEYS}


def extract_shape_feature(mask):
    """Scale-invariant shape / border descriptors.

    All features are ratios or normalized quantities, so they no longer depend on
    image resolution (this was the main cause of out-of-distribution predictions on
    down-sampled images)."""
    features = _zeros()
    binary_mask = (mask > 0).astype(np.uint8)
    total_px = float(binary_mask.size)
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return features

    contour = max(contours, key=cv.contourArea)
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    if area <= 0:
        return features

    # A. size relative to the whole image (scale-invariant)
    features["area_ratio"] = area / total_px

    # B. border smoothness
    if perimeter > 0:
        features["circularity"] = 4 * np.pi * area / (perimeter ** 2)

    if len(contour) >= 5:
        (_, _), (Ma, ma), _ = cv.fitEllipse(contour)
        if ma > 0:
            features["eccentricity"] = float(np.sqrt(max(0.0, 1 - (Ma / ma) ** 2)))

    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    hull_perimeter = cv.arcLength(hull, True)
    features["solidity"] = area / hull_area if hull_area > 0 else 0.0
    # >1 means the real border is more "wrinkled" than its convex hull
    features["border_irregularity"] = perimeter / hull_perimeter if hull_perimeter > 0 else 0.0

    x, y, w, h = cv.boundingRect(contour)
    features["extent"] = area / float(w * h) if w * h > 0 else 0.0
    return features
