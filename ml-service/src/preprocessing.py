import cv2 as cv

# Processing before Segmentation

def preprocess_denoise(image_rgb):
    image = cv.medianBlur(image_rgb, 5)
    return image

# Processing post segmentation

def preprocess_postsegment(image_rgb, mask, use_clahe=True):
    lesion = cv.bitwise_and(image_rgb, image_rgb, mask=mask)
    lab = cv.cvtColor(lesion, cv.COLOR_RGB2LAB)
    l, a, b = cv.split(lab)
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
    lab = cv.merge((l, a, b))
    preprocessed_image = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    preprocessed_image = cv.bilateralFilter(preprocessed_image, 7, 50, 50)
    return preprocessed_image
