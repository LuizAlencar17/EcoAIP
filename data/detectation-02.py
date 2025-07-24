import cv2
import ast
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import CoarseDropout


class DetectionDataset(Dataset):
    """
    Custom PyTorch Dataset for object detection using YOLO format.

    - Supports hybrid training with synthetic image degradation.
    - Can handle bounding boxes in pixel or normalized format.
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
        # Load and sample n rows from CSV containing image paths and detections
        self.data = (
            pd.read_csv(csv_file).sample(n=n, random_state=seed).reset_index(drop=True)
        )
        self.img_size = img_size[0]  # Assumes square input (e.g., 640x640 → 640)
        self.is_train = is_train
        self.hybrid_prob = (
            hybrid_prob if self.is_train else 0
        )  # Only use synthetic images in training
        self.bbox_is_normalized = bbox_is_normalized

        # Define synthetic degradation pipeline for simulating low-quality conditions
        self.synthetic_transform = A.Compose(
            [
                A.RandomGamma(gamma_limit=(30, 170), p=1.0),  # Random lighting/exposure
                A.OneOf(  # Random blur: simulates different types of distortion
                    [
                        A.Blur(blur_limit=7),
                        A.GaussianBlur(blur_limit=(3, 7)),
                        A.MotionBlur(blur_limit=7),
                    ],
                    p=0.5,
                ),
            ]
        )

        # Define standard augmentation pipeline for training and validation
        bbox_format = "yolo"  # Expected format: [cx, cy, w, h], all normalized
        if self.is_train:
            self.augs = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    # Use conservative Affine without newer args
                    A.Affine(
                        scale=1.0,
                        translate_percent=0.05,
                        rotate=15,
                        shear=5,
                        p=0.7,
                    ),
                    A.RandomScale(scale_limit=0.2, p=0.4),
                    # PadIfNeeded with older API
                    A.PadIfNeeded(
                        min_height=self.img_size,
                        min_width=self.img_size,
                        border_mode=0,
                        p=1.0,
                    ),
                    # Light/color augmentation
                    A.CLAHE(clip_limit=2.0, p=0.3),
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.5),
                    A.RGBShift(p=0.3),
                    A.ChannelShuffle(p=0.1),
                    # Drop the unsupported 'mean' and use only var_limit as float
                    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                    # Compression fallback: no args, just default compression
                    A.ImageCompression(p=0.2),
                    A.OneOf(
                        [
                            A.MotionBlur(p=0.2),
                            A.MedianBlur(blur_limit=5, p=0.1),
                            A.GaussianBlur(blur_limit=5, p=0.1),
                        ],
                        p=0.3,
                    ),
                    # Use GridDropout instead of Cutout/CoarseDropout
                    A.GridDropout(ratio=0.4, p=0.5),
                    A.Resize(height=self.img_size, width=self.img_size, p=1.0),
                    A.Normalize(
                        mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0
                    ),
                    ToTensorV2(p=1.0),
                ],
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_visibility=0.1,
                ),
            )

        else:
            # No heavy augmentation for validation/testing
            self.augs = A.Compose(
                [
                    A.Resize(height=self.img_size, width=self.img_size, p=1.0),
                    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
                    ToTensorV2(p=1.0),
                ],
                bbox_params=A.BboxParams(
                    format=bbox_format, label_fields=["class_labels"]
                ),
            )

    def __len__(self):
        return len(self.data)  # Return total number of samples

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
            img = cv2.imread(row["path"])  # Read image using OpenCV
            if img is None:
                # Skip image if it can't be read
                print(f"Warning: Could not read image: {row['path']}. Skipping.")
                return torch.empty((3, self.img_size, self.img_size)), torch.zeros(
                    (0, 5), dtype=torch.float32
                )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        except Exception as e:
            print(f"Error loading image {row['path']}: {e}")
            return torch.empty((3, self.img_size, self.img_size)), torch.zeros(
                (0, 5), dtype=torch.float32
            )

        # Apply synthetic transformation with some probability during training
        if self.is_train and random.random() < self.hybrid_prob:
            img = self.synthetic_transform(image=img)["image"]

        detections = ast.literal_eval(
            row["detections"]
        )  # Parse detection list from CSV
        bboxes = []
        class_ids = []
        h0, w0 = img.shape[:2]  # Original image height and width

        for det in detections:
            class_id = row["category"]  # Single class ID for all boxes in row

            # Convert bounding boxes to normalized YOLO format
            if not self.bbox_is_normalized:
                # Convert from pixel format [x, y, w, h] to [cx, cy, w, h]
                x_pixel, y_pixel, w_pixel, h_pixel = det["bbox"]
                if w_pixel <= 0 or h_pixel <= 0:
                    continue
                cx = (x_pixel + w_pixel / 2) / w0
                cy = (y_pixel + h_pixel / 2) / h0
                nw = w_pixel / w0
                nh = h_pixel / h0
            else:
                # Convert from [x_min, y_min, x_max, y_max] to [cx, cy, w, h]
                x_min_n, y_min_n, x_max_n, y_max_n = det["bbox"]
                nw = x_max_n - x_min_n
                nh = y_max_n - y_min_n
                cx = x_min_n + (nw / 2)
                cy = y_min_n + (nh / 2)

            # Clamp values to [0, 1] to avoid floating-point overflow
            cx, cy, nw, nh = map(lambda x: max(0.0, min(1.0, x)), [cx, cy, nw, nh])
            if nw <= 0 or nh <= 0:
                continue

            bboxes.append([cx, cy, nw, nh])
            class_ids.append(class_id)

        # Apply augmentations and get transformed image + bboxes
        try:
            transformed = self.augs(image=img, bboxes=bboxes, class_labels=class_ids)
        except ValueError as e:
            # Augmentation error (e.g., no visible bboxes left)
            print(f"Albumentations error on image {row['path']}: {e}")
            print(f"--> BBoxes sent: {bboxes}")
            return torch.empty((3, self.img_size, self.img_size)), torch.zeros(
                (0, 5), dtype=torch.float32
            )

        img_resized = transformed["image"]
        transformed_bboxes = transformed["bboxes"]

        # Combine class labels and bboxes into one tensor: [class_id, cx, cy, w, h]
        labels = []
        for bbox, class_label in zip(transformed_bboxes, transformed["class_labels"]):
            labels.append([class_label] + list(bbox))

        labels = torch.as_tensor(labels, dtype=torch.float32)
        if labels.shape[0] == 0:
            # Ensure shape is [0, 5] even if no detections remain
            labels = torch.zeros((0, 5), dtype=torch.float32)

        return img_resized, labels
