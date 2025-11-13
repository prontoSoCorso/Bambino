''' Configuration file for the project'''

import os
import torch
import random
import numpy as np

# Rileva il percorso della cartella "cellPIV" in modo dinamico
current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(current_file_path)
while os.path.basename(PROJECT_ROOT) != "Bambino":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

# Supported dataset subfolders
DATASET_SUBDIRS = {
    'raw': 'raw',
    'preprocessed': 'preprocessed',
    'normalized': 'normalized'
}

class settings:
    # General settings
    num_classes     = 2
    seed            = 2025
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu       = torch.cuda.device_count() > 1  # Variabile per controllare l'uso di più GPU
    hertz           = 30

    # Base folder under PROJECT_ROOT/data
    DATASETS_ROOT = os.path.join(PROJECT_ROOT, 'data')
    
    @staticmethod
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_filename      = "training_set.pt"
    validation_filename = "validation_set.pt"
    test_filename       = "test_set.pt"

    modality_dims = {
        "g": 8,
        "h": 13,
        "f": 17
        }
    
    num_features = sum(modality_dims.values())

    @staticmethod
    def get_dataset_path(dataset_type: str, filename: str = None) -> str:
        """
        Return the path to the desired dataset subfolder or file.

        Args:
            dataset_type: one of 'raw', 'preprocessed', or 'normalized'
            filename: optional filename (e.g. 'training_set.pt')

        Returns:
            Full path to the subfolder or file.

        Raises:
            KeyError: if dataset_type is not recognized.
        """
        try:
            subdir = DATASET_SUBDIRS[dataset_type]
        except KeyError:
            raise KeyError(f"Unknown dataset_type '{dataset_type}'. "
                           f"Choose from {list(DATASET_SUBDIRS.keys())}")

        base_path = os.path.join(settings.DATASETS_ROOT, subdir)
        if filename:
            return os.path.join(base_path, filename)
        return base_path

    
class Config_02_normalization:
    input_dataset_type = "preprocessed"
    output_dataset_type = "normalized"

