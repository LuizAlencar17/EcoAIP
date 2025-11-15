import torch.optim as optim

from models.baselines.resnet50 import get_resnet50_classifier
from models.baselines.mobilenet_v2 import get_mobilenet_v2_classifier
from models.baselines.yolov11 import YOLOV11

from models.aip.aip_mobilenet_v2 import AIPMobileNetV2
from models.aip.aip_resnet50 import AIPResNet50
from models.aip.aip_yolo11 import AIPYolov11

from models.eco_aip.eco_aip_mobilenet_v2 import EcoAIPMobileNetV2
from models.eco_aip.eco_aip_resnet50 import EcoAIPResNet50
from models.eco_aip.eco_aip_yolo11 import EcoAIPYolov11

from models.zero_dce.zero_dce_resnet50 import ZeroDCEResNet50
from models.zero_dce.zero_dce_mobilenet_v2 import ZeroDCEMobileNetV2
from models.zero_dce.zero_dce_yolo11 import ZeroDCEYolov11

from models.ia.ia_resnet50 import IAResNet50
from models.ia.ia_yolo11 import IAYolov11
from models.ia.ia_mobilenet_v2 import IAMobileNetV2


def get_model(config, device, num_classes: int = 1):

    mapper = {
        # Yolos
        "Yolov11": YOLOV11,
        "AIPYolov11": AIPYolov11,
        "ZeroDCEYolov11": ZeroDCEYolov11,
        "IAYolov11": IAYolov11,
        "EcoAIPYolov11": EcoAIPYolov11,
        # Resnets
        "ResNet50": get_resnet50_classifier,
        "AIPResNet50": AIPResNet50,
        "EcoAIPResNet50": EcoAIPResNet50,
        "ZeroDCEResNet50": ZeroDCEResNet50,
        "IAResNet50": IAResNet50,
        # MobileNets
        "MobileNetV2": get_mobilenet_v2_classifier,
        "ZeroDCEMobileNetV2": ZeroDCEMobileNetV2,
        "IAMobileNetV2": IAMobileNetV2,
        "AIPMobileNetV2": AIPMobileNetV2,
        "EcoAIPMobileNetV2": EcoAIPMobileNetV2,
    }
    model = mapper[config.MODEL](num_classes)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_MODEL)
    return model.to(device), optimizer
