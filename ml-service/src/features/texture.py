import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# Uniform LBP with P=8 -> 10 possible patterns
LBP_P = 8
LBP_R = 1
LBP_BINS = LBP_P + 2  # uniform patterns: 0..P plus the non-uniform bucket
LBP_KEYS = [f"lbp_{i}" for i in range(LBP_BINS)]
GLCM_KEYS = ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
EXTRA_KEYS = ["edge_density", "intensity_entropy"]
TEXTURE_KEYS = LBP_KEYS + GLCM_KEYS + EXTRA_KEYS


def _zeros():
    return {k: 0.0 for k in TEXTURE_KEYS}


def extract_texture_features(image_rgb, mask):
    """Rotation-invariant uniform LBP + GLCM (Haralick) + edge density + entropy.

    These capture the structural/pattern complexity inside the lesion far better than
    the original 8-neighbour count histogram."""
    features = _zeros()
    lesion = mask > 0
    if int(np.sum(lesion)) < 10:
        return features

    gray = cv.cvtColor(image_rgb, cv.COLOR_RGB2GRAY)
    # Local contrast enhancement helps texture descriptors (kept INSIDE texture only)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # --- Uniform LBP histogram over lesion pixels ---
    lbp = local_binary_pattern(gray_eq, LBP_P, LBP_R, method="uniform")
    lbp_vals = lbp[lesion]
    hist, _ = np.histogram(lbp_vals, bins=np.arange(0, LBP_BINS + 1), density=True)
    for i in range(LBP_BINS):
        features[f"lbp_{i}"] = float(hist[i])

    # --- GLCM (Haralick) on the bounding box of the lesion, masked background to 0 ---
    ys, xs = np.where(lesion)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = gray_eq[y0:y1, x0:x1].copy()
    patch_mask = lesion[y0:y1, x0:x1]
    patch[~patch_mask] = 0
    levels = 32
    q = (patch.astype(np.uint16) * (levels - 1) // 255).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 2],
                        levels=levels, symmetric=True, normed=True)
    features["glcm_contrast"] = float(graycoprops(glcm, "contrast").mean())
    features["glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity").mean())
    features["glcm_energy"] = float(graycoprops(glcm, "energy").mean())
    features["glcm_correlation"] = float(graycoprops(glcm, "correlation").mean())

    # --- Edge density (Sobel magnitude) inside the lesion ---
    gx = cv.Sobel(gray_eq, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray_eq, cv.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    features["edge_density"] = float(np.mean(mag[lesion]))

    # --- Intensity entropy inside the lesion ---
    vals = gray_eq[lesion]
    h, _ = np.histogram(vals, bins=32, range=(0, 256), density=True)
    h = h[h > 0]
    features["intensity_entropy"] = float(-np.sum(h * np.log2(h))) if h.size else 0.0
    return features
