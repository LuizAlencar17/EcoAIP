DATA_TRAIN_CSV_PATH = (
    "/data/luiz/dataset/partitions/detection-classifier/caltech/train.csv"
)
DATA_VAL_CSV_PATH = "/data/luiz/dataset/partitions/detection-classifier/caltech/val.csv"
DATA_TEST_CSV_PATH = (
    "/data/luiz/dataset/partitions/detection-classifier/caltech/test.csv"
)
MODEL = "Yolov11"
OUTPUT_DIR = "/data/luiz/dataset/EcoAIP/caltech"
PATIENCE = 20
NUM_CLASS = 1
TRAIN_SIZE = 10
VAL_SIZE = 6
TEST_SIZE = 10
BATCH_SIZE = 4
BBOX_IS_NORMALIZED = False
EPOCHS = 2
TASK = "detection-classifier"
LEARNING_RATE_MODEL = 1e-4
IMAGE_SIZE = (640, 640)
SEED = 42
