DATA_TRAIN_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier/serengeti/train.csv"
)
DATA_VAL_CSV_PATH = "/data/luiz/dataset/partitions/species-classifier/serengeti/val.csv"
DATA_TEST_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier/serengeti/test.csv"
)
MODEL = "AIPMobileNetV2"
OUTPUT_DIR = "/data/luiz/dataset/EcoAIP/serengeti"
PATIENCE = 20
NUM_CLASS = 10
TRAIN_SIZE = 4800
VAL_SIZE = 600
TEST_SIZE = 1000
BATCH_SIZE = 8
BBOX_IS_NORMALIZED = False
EPOCHS = 100
TASK = "species-classifier"
LEARNING_RATE_MODEL = 1e-4
IMAGE_SIZE = (224, 224)
SEED = 42
