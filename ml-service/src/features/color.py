import cv2 as cv
import numpy as np

COLOR_KEYS = ["l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std",
              "hue_std", "sat_mean", "sat_std", "dark_fraction", "l_range"]


def _zeros():
    return {k: 0.0 for k in COLOR_KEYS}


def extract_color(image_rgb, mask):
    """Color statistics computed on the ORIGINAL (denoised) RGB lesion.

    NOTE: color is intentionally measured BEFORE CLAHE/contrast enhancement, because
    absolute color/brightness and color variegation are diagnostic (CLAHE would
    erase them). Adds HSV-based variegation (hue_std), saturation, the fraction of
    very dark pixels (proxy for black/blue-grey regions) and the L dynamic range."""
    features = _zeros()
    lesion = mask > 0
    n = int(np.sum(lesion))
    if n == 0:
        return features

    lab = cv.cvtColor(image_rgb, cv.COLOR_RGB2LAB)
    l = lab[:, :, 0][lesion].astype(np.float32)
    a = lab[:, :, 1][lesion].astype(np.float32)
    b = lab[:, :, 2][lesion].astype(np.float32)
    features["l_mean"] = float(np.mean(l))
    features["l_std"] = float(np.std(l))
    features["a_mean"] = float(np.mean(a))
    features["a_std"] = float(np.std(a))
    features["b_mean"] = float(np.mean(b))
    features["b_std"] = float(np.std(b))

    hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)
    hue = hsv[:, :, 0][lesion].astype(np.float32)
    sat = hsv[:, :, 1][lesion].astype(np.float32)
    features["hue_std"] = float(np.std(hue))      # color variegation
    features["sat_mean"] = float(np.mean(sat))
    features["sat_std"] = float(np.std(sat))

    # Dark-region fraction (black/blue-grey areas are a melanoma cue)
    features["dark_fraction"] = float(np.mean(l < 60.0))
    # Robust luminance range (multi-color lesions span a wider range)
    features["l_range"] = float(np.percentile(l, 95) - np.percentile(l, 5))
    return features
