import torch

# Configuration for conv film + metadata
dataset_type = 'normalized'

# model config
TARGET_LEN = 250          # target temporal length (pad/truncate to this)
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 50              # early stopping epochs on val balanced accuracy
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1

# Modalities to use (g,h,f as in your project)
MODALITIES = ['g', 'h', 'f']

# Numeric and categorical metadata fields (same as earlier pipeline)
numeric_features = ["age", "duration_s"]
cat_features = ["sex"]