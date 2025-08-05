DATA_TRAIN_CSV_PATH = (
    "/data/luiz/dataset/partitions/detection-classifier/serengeti/train.csv"
)
DATA_VAL_CSV_PATH = (
    "/data/luiz/dataset/partitions/detection-classifier/serengeti/val.csv"
)
DATA_TEST_CSV_PATH = (
    "/data/luiz/dataset/partitions/detection-classifier/serengeti/test.csv"
)
MODEL = "Yolov11"
OUTPUT_DIR = "/data/luiz/dataset/EcoAIP/serengeti"
PATIENCE = 20
NUM_CLASS = 1
TRAIN_SIZE = 4800
VAL_SIZE = 600
TEST_SIZE = 1000
BATCH_SIZE = 4
BBOX_IS_NORMALIZED = True
EPOCHS = 150
TASK = "detection-classifier"
LEARNING_RATE_MODEL = 1e-4
IMAGE_SIZE = (640, 640)
SEED = 42
