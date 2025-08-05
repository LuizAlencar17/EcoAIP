import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ClassificationDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        img_size: tuple = (224, 224),
        n: int = 1000,
        seed: int = 42,
        is_train: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.is_train = is_train

        # 1. Carrega e filtra o DataFrame para garantir que todos os arquivos existem
        df_initial = pd.read_csv(csv_file)
        print(f"Verificando {len(df_initial)} imagens listadas em '{csv_file}'...")

        # Remove linhas com caminhos de imagem ausentes para evitar erros
        original_len = len(df_initial)
        df_initial["path_exists"] = df_initial["path"].apply(os.path.exists)
        self.data_frame = df_initial[df_initial["path_exists"]].copy()

        if len(self.data_frame) < original_len:
            print(
                f"AVISO: {original_len - len(self.data_frame)} imagens não encontradas foram removidas."
            )

        # Amostragem e preparação final do DataFrame
        self.data_frame = self.data_frame.sample(
            n=min(n, len(self.data_frame)), random_state=seed
        ).reset_index(drop=True)
        self.data_frame["label"] = self.data_frame["category"]

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Pipeline de validação/teste sem augmentations aleatórias
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int):
        img_path = self.data_frame.iloc[idx]["path"]
        image = Image.open(img_path).convert("RGB")
        label = self.data_frame.iloc[idx]["label"]

        # 3. Aplica a transformação correta baseada no modo (treino ou validação)
        if self.is_train:
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)

        return image, label
