import torch

# Configuration for conv film + metadata
dataset_type = 'augmented_normalized'

# model config
TARGET_LEN = 250          # target temporal length (pad/truncate to this)
BATCH_SIZE = 16
NUM_EPOCHS = 40
LR = 5e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 10              # early stopping epochs on val balanced accuracy
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1

# Modalities to use (gaze, head, face)
MODALITIES = ['g', 'h', 'f']

# Numeric and categorical metadata fields
numeric_features = ["age"]
cat_features = ["sex"]

# FiLM and CNN model parameters
channels=(8, 16)
kernel_size=7
dropout=0.7
head_dim=16