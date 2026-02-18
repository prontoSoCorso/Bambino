import torch

# Dataset Config
dataset_type = 'augmented_normalized' # "normalized", "augmented_normalized"
training_filename = "train_data.pkl"
validation_filename = "val_data.pkl"
test_filename = "test_data.pkl"

# Model Choice
# Options: 'embeddings+histgb', 'pca+tabpfn', 'embeddings+mlp'
model_type = 'embeddings+mlp'

# MOMENT Config
moment_model_name = "AutonLab/MOMENT-1-large"
num_classes = 2         # Binary Classification
n_channels = 38         # Gaze(8) + Head(13) + Face(17)
target_len = 512        # MOMENT expects 512 time steps
batch_size = 8          # Keep small for MOMENT inference

# Classifier Config
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")