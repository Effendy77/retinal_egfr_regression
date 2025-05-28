import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class RetinalDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, target_col='egfr'):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.data.iloc[idx][self.target_col], dtype=torch.float32)
        return image, label
