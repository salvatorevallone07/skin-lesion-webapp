import cv2 as cv
import numpy as np

def segment_lesion(image_rgb):
    image_hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)
    saturation = image_hsv[:, :, 1]
    # Use Otsu for the large dataset
    _, mask = cv.threshold(saturation, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Remove noise of skin, regular edges
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        mask_clean = np.zeros_like(mask)
        mask_clean[labels == largest_label] = 255
        mask = mask_clean
    return mask
