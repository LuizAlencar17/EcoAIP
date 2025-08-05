DATA_TRAIN_CSV_PATH = (
    "/data/luiz/dataset/partitions/detection-classifier/nacti/train.csv"
)
DATA_VAL_CSV_PATH = "/data/luiz/dataset/partitions/detection-classifier/nacti/val.csv"
DATA_TEST_CSV_PATH = "/data/luiz/dataset/partitions/detection-classifier/nacti/test.csv"
MODEL = "EcoAIPYolov11"
OUTPUT_DIR = "/data/luiz/dataset/EcoAIP/nacti"
PATIENCE = 20
NUM_CLASS = 1
TRAIN_SIZE = 1200
VAL_SIZE = 600
TEST_SIZE = 600
BATCH_SIZE = 4
BBOX_IS_NORMALIZED = False
EPOCHS = 150
TASK = "detection-classifier"
LEARNING_RATE_MODEL = 1e-4
IMAGE_SIZE = (640, 640)
SEED = 42
