DATA_TRAIN_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier-cropped/wcs/train.csv"
)
DATA_VAL_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier-cropped/wcs/val.csv"
)
DATA_TEST_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier-cropped/wcs/test.csv"
)
MODEL = "MobileNetV2"
OUTPUT_DIR = "/data/luiz/dataset/EcoAIP/wcs"
PATIENCE = 20
NUM_CLASS = 10
TRAIN_SIZE = 4800
VAL_SIZE = 600
TEST_SIZE = 1000
BATCH_SIZE = 8
BBOX_IS_NORMALIZED = False
EPOCHS = 100
TASK = "species-classifier-cropped"
LEARNING_RATE_MODEL = 1e-4
IMAGE_SIZE = (224, 224)
SEED = 42
