DATA_TRAIN_CSV_PATH = (
    "/data/luiz/dataset/partitions/species-classifier/caltech/train.csv"
)
DATA_VAL_CSV_PATH = "/data/luiz/dataset/partitions/species-classifier/caltech/val.csv"
DATA_TEST_CSV_PATH = "/data/luiz/dataset/partitions/species-classifier/caltech/test.csv"
MODEL = "ResNet50"
MODEL_NAME = "normal"
OUTPUT_DIR = "/data/luiz/dataset/EcoAIP/caltech"
PATIENCE = 15
NUM_CLASS = 10
TRAIN_SIZE = 1200
BATCH_SIZE = 16
AUGMENT_IMAGE = True
EPOCHS = 100
TASK = "species-classifier"
LEARNING_RATE_MODEL = 1e-4
IMAGE_SIZE = (400, 400)
SEED = 42
