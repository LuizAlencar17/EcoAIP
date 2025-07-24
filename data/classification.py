import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


# 1. Crie uma classe para sua lógica de augmentation
#    Isso encapsula a lógica da função em um formato que o transforms.Compose entende.
class CustomAugmentations:
    """Aplica uma série de augmentations com 50% de chance cada."""

    def __call__(self, x):
        # Ajuste de gamma
        if torch.rand(1).item() > 0.5:
            x = TF.adjust_gamma(x, gamma=torch.empty(1).uniform_(0.3, 2.0).item())

        # Blur gaussiano
        if torch.rand(1).item() > 0.5:
            # O kernel deve ser de ímpares
            x = TF.gaussian_blur(x, kernel_size=[3, 3])

        # Adição de ruído
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(x) * 0.05
            x = x + noise

        # Garante que os valores da imagem permaneçam no intervalo [0, 1]
        return torch.clamp(x, 0, 1)


# 2. Atualize a classe de Dataset para usar a augmentation
class ClassificationDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        img_size: tuple = (224, 224),
        n: int = 1000,
        seed: int = 42,
        is_train: bool = False,  # Adicionado parâmetro para controlar a augmentation
        **kwargs,
    ):
        self.data_frame = (
            pd.read_csv(csv_file).sample(n=n, random_state=seed).reset_index(drop=True)
        )
        self.data_frame["label"] = self.data_frame["category"]

        # Cria a lista de transformações base
        transform_list = [transforms.Resize(img_size), transforms.ToTensor()]

        if is_train:
            transform_list.append(CustomAugmentations())

        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int):
        img_path = self.data_frame.iloc[idx]["path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {img_path}. Retornando None.")
            return None, None  # Ou trate o erro como preferir

        label = self.data_frame.iloc[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label
