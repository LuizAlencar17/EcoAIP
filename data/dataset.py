import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional


class AnimalDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        transform: Optional[transforms.Compose] = None,
        n: int = 1000,
    ):
        self.data_frame = (
            pd.read_csv(csv_file).sample(n=n, random_state=42).reset_index(drop=True)
        )
        self.transform = transform
        self.data_frame["label"] = self.data_frame["category"]

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int):
        img_path = self.data_frame.iloc[idx]["path"]
        image = Image.open(img_path).convert("RGB")
        label = self.data_frame.iloc[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label
