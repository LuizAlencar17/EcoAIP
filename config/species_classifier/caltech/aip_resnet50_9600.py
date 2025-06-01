DATA_TRAIN_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier/caltech/train.csv"
)
DATA_VAL_CSV_PATH = "/data/luiz/dataset/partitions/species-classifier/caltech/val.csv"
DATA_TEST_CSV_PATH = "/data/luiz/dataset/partitions/species-classifier/caltech/test.csv"
MODEL_NAME = "normal"
MODEL = "AIPResNet50"
OUTPUT_DIR = "/data/luiz/dataset/EcoAIP/caltech"
PATIENCE = 8
NUM_CLASS = 10
TRAIN_SIZE = 9600
BATCH_SIZE = 16
AUGMENT_IMAGE = True
EPOCHS = 40
TASK = "species-classifier"
LEARNING_RATE_MODEL = 1e-4
IMAGE_SIZE = (224, 224)
SEED = 42
