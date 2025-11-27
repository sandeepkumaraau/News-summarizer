import torch


class Config:
    MODEL_NAME = "google/pegasus-cnn_dailymail"
    SEED = 42
    MAX_INPUT_LEN = 256
    MAX_TARGET_LEN = 128
    BATCH_SIZE = 2
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Data subsetting (for faster testing)
    TRAIN_SUBSET_RATIO = 0.01
    VAL_SUBSET_RATIO = 0.01
