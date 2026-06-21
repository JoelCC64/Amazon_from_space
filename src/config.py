import os

# Determine if running in Kaggle or local environment
KAGGLE_BASE_DIR = '/kaggle/input/datasets/nikitarom/planets-dataset/planet/planet'
LOCAL_BASE_DIR = os.environ.get('AMAZON_DATA_DIR', './data')

if os.path.exists(KAGGLE_BASE_DIR):
    RUTA_DATOS = KAGGLE_BASE_DIR
else:
    RUTA_DATOS = LOCAL_BASE_DIR

# Derived data paths
CSV_PATH = os.path.join(RUTA_DATOS, 'train_classes.csv')
IMG_PATH_TRAIN = os.path.join(RUTA_DATOS, 'train-jpg')
TEST_IMG_PATH = os.path.join(RUTA_DATOS, 'test-jpg')

# Default hyperparameters
NUM_CLASSES = 17
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DEFAULT_THRESHOLD = 0.5
IMG_SIZE = (224, 224)

# Normalization constants (ImageNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
