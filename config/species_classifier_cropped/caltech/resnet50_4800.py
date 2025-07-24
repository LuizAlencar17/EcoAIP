DATA_TRAIN_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier-cropped/caltech/train.csv"
)
DATA_VAL_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier-cropped/caltech/val.csv"
)
DATA_TEST_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier-cropped/caltech/test.csv"
)
MODEL = "ResNet50"
OUTPUT_DIR = "/data/luiz/dataset/EcoAIP/caltech"
PATIENCE = 20
NUM_CLASS = 10
TRAIN_SIZE = 48
VAL_SIZE = 16
TEST_SIZE = 16
BATCH_SIZE = 16
BBOX_IS_NORMALIZED = False
EPOCHS = 2
TASK = "species-classifier-cropped"
LEARNING_RATE_MODEL = 1e-4
IMAGE_SIZE = (224, 224)
SEED = 42
