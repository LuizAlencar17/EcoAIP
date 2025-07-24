import ast
import random

import cv2
import pandas as pd
import torch

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DetectionDataset(Dataset):
    """
    A dataset that implements hybrid training strategies and supports
    both pixel-based and normalized bounding boxes.

    This class generates low-quality synthetic images (e.g., light variations and blur)
    to randomly replace original images during training with a given probability,
    making the model more robust to adverse conditions [cite: 7, 24].
    """

    def __init__(
        self,
        csv_file: str,
        img_size: tuple,
        n: int = 1000,
        seed: int = 42,
        is_train: bool = False,
        hybrid_prob: float = 2 / 3,
        bbox_is_normalized: bool = False,
        **kwargs,
    ):
        self.data = (
            pd.read_csv(csv_file).sample(n=n, random_state=seed).reset_index(drop=True)
        )
        self.img_size = img_size[0]
        self.is_train = is_train
        self.hybrid_prob = hybrid_prob if self.is_train else 0
        self.bbox_is_normalized = bbox_is_normalized

        # --- Synthetic Image Pipeline (Low Quality Simulation) ---
        self.synthetic_transform = A.Compose(
            [
                A.RandomGamma(gamma_limit=(30, 170), p=1.0),
                A.OneOf(
                    [
                        A.Blur(blur_limit=7),
                        A.GaussianBlur(blur_limit=(3, 7)),
                        A.MotionBlur(blur_limit=7),
                    ],
                    p=0.5,
                ),
            ]
        )

        # --- Standard Augmentation Pipeline ---
        bbox_format = "yolo"

        if self.is_train:
            self.augs = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8
                    ),
                    A.Resize(height=self.img_size, width=self.img_size, p=1.0),
                    A.Normalize(
                        mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0
                    ),
                    ToTensorV2(p=1.0),
                ],
                bbox_params=A.BboxParams(
                    format=bbox_format,
                    label_fields=["class_labels"],
                    min_visibility=0.1,
                ),
            )
        else:
            self.augs = A.Compose(
                [
                    A.Resize(height=self.img_size, width=self.img_size, p=1.0),
                    A.Normalize(
                        mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0
                    ),
                    ToTensorV2(p=1.0),
                ],
                bbox_params=A.BboxParams(
                    format=bbox_format, label_fields=["class_labels"]
                ),
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        try:
            img = cv2.imread(row["path"])
            if img is None:
                print(f"Warning: Could not read image: {row['path']}. Skipping.")
                return torch.empty((3, self.img_size, self.img_size)), torch.zeros(
                    (0, 5), dtype=torch.float32
                )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {row['path']}: {e}")
            return torch.empty((3, self.img_size, self.img_size)), torch.zeros(
                (0, 5), dtype=torch.float32
            )

        if self.is_train and random.random() < self.hybrid_prob:
            img = self.synthetic_transform(image=img)["image"]

        detections = ast.literal_eval(row["detections"])
        bboxes = []
        class_ids = []
        h0, w0 = img.shape[:2]

        for det in detections:
            class_id = row["category"]
            class_id = 0

            if not self.bbox_is_normalized:
                x_pixel, y_pixel, w_pixel, h_pixel = det["bbox"]
                if w_pixel <= 0 or h_pixel <= 0:
                    continue
                cx = (x_pixel + w_pixel / 2) / w0
                cy = (y_pixel + h_pixel / 2) / h0
                nw = w_pixel / w0
                nh = h_pixel / h0
            else:
                x_min_n, y_min_n, x_max_n, y_max_n = det["bbox"]
                nw = x_max_n - x_min_n
                nh = y_max_n - y_min_n
                cx = x_min_n + (nw / 2)
                cy = y_min_n + (nh / 2)

            cx, cy, nw, nh = map(lambda x: max(0.0, min(1.0, x)), [cx, cy, nw, nh])

            if nw <= 0 or nh <= 0:
                continue

            bboxes.append([cx, cy, nw, nh])
            class_ids.append(class_id)

        try:
            transformed = self.augs(image=img, bboxes=bboxes, class_labels=class_ids)
        except ValueError as e:
            print(f"Albumentations error on image {row['path']}: {e}")
            print(f"--> BBoxes sent: {bboxes}")
            return torch.empty((3, self.img_size, self.img_size)), torch.zeros(
                (0, 5), dtype=torch.float32
            )

        img_resized = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        labels = []

        for bbox, class_label in zip(transformed_bboxes, transformed["class_labels"]):
            labels.append([class_label] + list(bbox))

        labels = torch.as_tensor(labels, dtype=torch.float32)
        if labels.shape[0] == 0:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        return img_resized, labels
