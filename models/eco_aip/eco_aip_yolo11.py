import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

# from models.aip.aip_improved_modules import ZeroDCE_PlusPlus

# from models.aip.aip_fast_modules import FastNLPP as EnhancedNLPP, FastDIP as EnhancedDIP
from ultralytics.utils.torch_utils import initialize_weights


class EcoAIPYolov11(nn.Module):
    """
    Custom YOLOv11 model wrapper that allows replacement of the detection head for fine-tuning
    with a different number of output classes.
    """

    def __init__(
        self,
        num_classes,
        model_name="/data/luiz/dataset/models/original-yolos/yolo11s.pt",
        multi_scale_training=True,
    ):
        """
        Initializes the YOLOv11 model with a new detection head for the specified number of classes.

        Args:
            num_classes (int): Number of output classes.
            model_name (str): Path to the pre-trained YOLOv11 weights.
            multi_scale_training (bool): Whether to use multi-scale training.
        """
        super().__init__()
        # self.enhancer = ZeroDCE_PlusPlus()

        # --- Weight File Sanity Check ---
        weights_file = Path(model_name).resolve()
        if not weights_file.is_file():
            raise FileNotFoundError(
                f"Weight file not found: {weights_file}. "
                f"Download the appropriate pre-trained weights before instantiating the model."
            )

        # Load the pre-trained YOLO model from Ultralytics
        yolo_pretrain = YOLO(weights_file)
        self.model = yolo_pretrain.model

        # This technique involves changing the input image size every few
        # training iterations to make the model robust to different resolutions.
        self.multi_scale_training = multi_scale_training

        # Get the original detection head
        original_head = self.model.model[-1]

        # Check if the last layer is actually a Detect head
        if not isinstance(original_head, Detect):
            raise TypeError(
                "The last layer of the loaded model is not a 'Detect' detection head."
            )

        # Get the input channels of the original detection head
        in_channels = [m[0].conv.in_channels for m in original_head.cv2]

        # Create a NEW detection head with the correct number of classes
        new_head = Detect(nc=num_classes, ch=in_channels)

        # Transplant essential metadata from the old head to the new one
        new_head.f = original_head.f
        new_head.i = original_head.i
        new_head.type = original_head.type
        new_head.stride = original_head.stride

        # Initialize the weights of the new head robustly
        new_head.apply(initialize_weights)
        new_head.bias_init()

        # Replace the old head with the new one in the model
        self.model.model[-1] = new_head

    def forward(self, x, *args, **kwargs):
        """
        x: [B,3,H,W] em [0,1].
        Retorna exatamente o que o YOLO retornaria (lista/tensor de predições).
        """
        x = x.float().clamp(0.0, 1.0)

        # A DCE já é estável em AMP; se seu trainer usa autocast, o grad flui normal.
        y, params = self.enhancer(x)  # y em [0,1]
        gate = params["gate"]  # [B,1,H,W]
        x = torch.clamp(x + gate * (y - x), 0, 1)

        # x = y  # usa a imagem totalmente melhorada

        # IMPORTANTÍSSIMO: retornar apenas as SAÍDAS do YOLO
        return self.model(x, *args, **kwargs)
