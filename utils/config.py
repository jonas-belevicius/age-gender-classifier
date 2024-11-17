import os

project_dir = os.path.abspath("../")

RANDOM_SEED = 1
DATA_DIR = os.path.join(project_dir, "data")
TRAIN_VAL_DIR = os.path.join(DATA_DIR, "train_validation_data")
TRAIN_EPOCHS = 5
FINE_TUNE_EPOCHS = 10
TRAIN_LR = 1e-3
FINE_TUNE_LR = 1e-5
BATCH_SIZE = 8
NUM_CLASSES = 2
CHECKPOINT_PATH = os.path.join(project_dir, "model_checkpoints")
LOGS_PATH = os.path.join(project_dir, "training_logs")
LOGS_NAME = "resnet18"
