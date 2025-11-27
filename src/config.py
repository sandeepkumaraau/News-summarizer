import torch


class Config:
    MODEL_NAME = "google/pegasus-cnn_dailymail"
    SEED = 42
    MAX_INPUT_LEN = 128
    MAX_TARGET_LEN = 56
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Data subsetting (for faster testing)
    TRAIN_SUBSET_RATIO = 0.005
    VAL_SUBSET_RATIO = 0.01
