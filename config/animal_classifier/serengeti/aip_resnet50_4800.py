DATA_TRAIN_CSV_PATH = (
    "/data/luiz/dataset/partitions/animal-classifier/serengeti/train.csv"
)
DATA_VAL_CSV_PATH = "/data/luiz/dataset/partitions/animal-classifier/serengeti/val.csv"
DATA_TEST_CSV_PATH = (
    "/data/luiz/dataset/partitions/animal-classifier/serengeti/test.csv"
)
MODEL_NAME = "normal"
MODEL = "AIPResNet50"
OUTPUT_DIR = "/data/luiz/dataset/EcoAIP/serengeti"
PATIENCE = 8
NUM_CLASS = 10
TRAIN_SIZE = 4800
BATCH_SIZE = 16
AUGMENT_IMAGE = True
EPOCHS = 40
TASK = "animal-classifier"
LEARNING_RATE_MODEL = 1e-4
IMAGE_SIZE = (224, 224)
SEED = 42
