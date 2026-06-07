import numpy as np
from src.features.shape import extract_shape_feature
from src.features.asymmetry import extract_asymmetry
from src.features.color import extract_color
from src.features.texture import extract_texture_features


def extract_features(image_rgb, mask):
    """Build the full feature vector for one lesion.

    `image_rgb` is the DENOISED RGB image (no CLAHE applied at this stage); texture
    handles its own local enhancement internally. Returns the feature names and a
    float32 vector, in a stable order."""
    shape_features = extract_shape_feature(mask)
    asym_features = extract_asymmetry(mask)
    color_features = extract_color(image_rgb, mask)
    texture_feats = extract_texture_features(image_rgb, mask)
    features = {**shape_features, **asym_features, **color_features, **texture_feats}
    feature_names = list(features.keys())
    feature_values = np.array(list(features.values()), dtype=np.float32)
    feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_names, feature_values
