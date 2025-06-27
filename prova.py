''' main_preprocessing.py '''
import os, sys

# Configurazione dei percorsi
parent_dir = os.path.dirname(os.path.abspath(__file__))
while not os.path.basename(parent_dir) == "Bambino":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_01_preprocessing, utils
from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.BoaOpenFaceDataset import BoaOpenFaceDataset

def main(split_name: str, cfg: Config_01_preprocessing):
    """
    Carica, preprocessa e salva il dataset per uno split (train, validation, test)
    """
    # 1) Path di input e output
    in_path  = utils.get_dataset_path(cfg.input_dataset_type, getattr(utils, f"{split_name}_filename"))
    out_path = utils.get_dataset_path(cfg.output_dataset_type, getattr(utils, f"{split_name}_filename"))

    # 2) Carica il dataset raw
    ds = BoaOpenFaceDataset.load_dataset(in_path, modalities=None)

    print("ciao ciao ciao!")


if __name__ == "__main__":
    # Imposta seed
    utils.seed_everything(utils.seed)

    # Configurazione preprocess
    cfg = Config_01_preprocessing

    # Processa train, validation e test
    for split in ["train", "validation", "test"]:
        main(split, cfg)