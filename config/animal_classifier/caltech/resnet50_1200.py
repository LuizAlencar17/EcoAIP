DATA_TRAIN_CSV_PATH = (
    "/data/luiz/dataset/partitions/animal-classifier/caltech/train.csv"
)
DATA_VAL_CSV_PATH = "/data/luiz/dataset/partitions/animal-classifier/caltech/val.csv"
DATA_TEST_CSV_PATH = "/data/luiz/dataset/partitions/animal-classifier/caltech/test.csv"
MODEL = "ResNet50"
MODEL_NAME = "normal"
OUTPUT_DIR = "/data/luiz/dataset/EcoAIP/caltech"
PATIENCE = 8
NUM_CLASS = 2
TRAIN_SIZE = 1200
BATCH_SIZE = 16
AUGMENT_IMAGE = True
EPOCHS = 40
TASK = "animal-classifier"
LEARNING_RATE_MODEL = 1e-4
IMAGE_SIZE = (224, 224)
SEED = 42
