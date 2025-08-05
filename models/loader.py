import torch.optim as optim
from models.resnet import get_resnet50_classifier
from models.aip_resnet50 import AIPResNet50
from models.yolov11 import YOLOV11
from models.eco_aip_resnet50 import EcoAIPResNet50
from models.aip_yolo11 import AIPYolov11
from models.eco_aip_yolo11 import EcoAIPYolov11


def get_model(config, device, num_classes: int = 1):
    if config.MODEL == "Yolov11":
        model = YOLOV11(num_classes)

    elif config.MODEL == "AIPYolov11":
        model = AIPYolov11(num_classes)

    elif config.MODEL == "EcoAIPYolov11":
        model = EcoAIPYolov11(num_classes)

    elif config.MODEL == "ResNet50":
        model = get_resnet50_classifier(num_classes)

    elif config.MODEL == "AIPResNet50":
        model = AIPResNet50(num_classes)

    elif config.MODEL == "EcoAIPResNet50":
        model = EcoAIPResNet50(num_classes)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_MODEL)
    return model.to(device), optimizer
