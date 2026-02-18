import torch

# Configuration for InceptionTime + Metadata
dataset_type = 'normalized' # 'normalized' or 'augmented_normalized'

# Model Config
TARGET_LEN = 250          
BATCH_SIZE = 16
NUM_EPOCHS = 50      
PATIENCE = 10            
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1

# Modalities
MODALITIES = ['g', 'h', 'f']

# Metadata
numeric_features = ["age"]
cat_features = ["sex"]

# InceptionTime Hyperparameters
nb_filters = 16           
use_residual = True
depth = 3                
kernel_size = 39           # Small kernel (Odd number = No padding errors)
bottleneck_size = 8
dropout = 0.6           

# Training Config
LR = 1e-3
WEIGHT_DECAY = 0.01