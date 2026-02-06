import torch
import os

# Paths
BASE_DIR = "/home/phd2/Scrivania/CorsoRepo/Bambino/_03_train/moment_anomaly_detection"
os.makedirs(BASE_DIR, exist_ok=True)

# Model Config
MOMENT_MODEL = "AutonLab/MOMENT-1-large"
N_CHANNELS = 38
TARGET_LEN = 512 # MOMENT fixed input
BATCH_SIZE = 16  # Slightly higher for reconstruction
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anomaly Detection Config
NUM_CLASSES = 2  # Normal vs Anomaly
NORMAL_CLASS = 1  # Stimulus (Majority)
ANOMALY_CLASS = 0 # Control (Minority)
TRAIN_SPLIT_RATIO = 0.75 # 75% of Normal class used for training

# Fine-tuning Config
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 6
SCH_FACTOR = 0.5
MASK_RATIO = 0.3 # Masking ratio for training (MAE style)
SEED = 42