DATA_TRAIN_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier-cropped/nacti/train.csv"
)
DATA_VAL_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier-cropped/nacti/val.csv"
)
DATA_TEST_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier-cropped/nacti/test.csv"
)
MODEL = "MobileNetV2"
OUTPUT_DIR = "/data/luiz/dataset/EcoAIP/nacti"
PATIENCE = 20
NUM_CLASS = 10
TRAIN_SIZE = 1200
VAL_SIZE = 600
TEST_SIZE = 600
BATCH_SIZE = 8
BBOX_IS_NORMALIZED = False
EPOCHS = 100
TASK = "species-classifier-cropped"
LEARNING_RATE_MODEL = 1e-4
IMAGE_SIZE = (224, 224)
SEED = 42
