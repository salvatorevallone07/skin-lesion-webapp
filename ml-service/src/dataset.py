import os
import cv2 as cv
import pandas as pd
from torch.utils.data import Dataset

class ISIC2019Dataset(Dataset):
    """Loads the ISIC 2019 training set, keeping only MEL and NV samples.
    NV -> 0 (melanocytic nevus), MEL -> 1 (melanoma)."""

    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(csv_file)
        df = df[(df["MEL"] == 1) | (df["NV"] == 1)].copy()
        df["label"] = df["MEL"].apply(lambda x: 1 if x == 1 else 0)
        self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row["image"]
        label = int(row["label"])
        image_path = os.path.join(self.root_dir, image_id + ".jpg")
        image = cv.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
