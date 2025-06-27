import os, sys
import pickle
import numpy as np

# Configurazione dei percorsi
parent_dir = os.path.dirname(os.path.abspath(__file__))
while not os.path.basename(parent_dir) == "Bambino":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_02_normalization, utils
from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.BoaOpenFaceDataset import BoaOpenFaceDataset


def normalize_dataset(split_name: str, cfg: Config_02_normalization):
    """
    Carica, normalizza per canale e per pt_id, e salva il dataset normalizzato.
    """
    # Path di input e output
    in_path  = utils.get_dataset_path(cfg.input_dataset_type, getattr(utils, f"{split_name}_filename"))
    out_path = utils.get_dataset_path(cfg.output_dataset_type, getattr(utils, f"{split_name}_filename"))

    # Carica dataset preprocessato
    ds = BoaOpenFaceDataset.load_dataset(in_path, modalities=None)

    # Raggruppa dati per pt_id
    pt_ids = {inst.pt_id for inst in ds.instances}
    stats = {}
    for pt in pt_ids:
        # Estrai tutti i dati per quel paziente
        gaze_all = np.vstack([inst.gaze_info for inst in ds.instances if inst.pt_id == pt])
        head_all = np.vstack([inst.head_info for inst in ds.instances if inst.pt_id == pt])
        face_all = np.vstack([inst.face_info for inst in ds.instances if inst.pt_id == pt])
        # Calcola media e std per feature
        stats[pt] = {
            'g': (gaze_all.mean(axis=0), gaze_all.std(axis=0, ddof=0)),
            'h': (head_all.mean(axis=0), head_all.std(axis=0, ddof=0)),
            'f': (face_all.mean(axis=0), face_all.std(axis=0, ddof=0)),
        }

    # Applica normalizzazione per istanza
    for inst in ds.instances:
        mu_g, sigma_g = stats[inst.pt_id]['g']
        mu_h, sigma_h = stats[inst.pt_id]['h']
        mu_f, sigma_f = stats[inst.pt_id]['f']
        # Evita divisione per zero
        inst.gaze_info = (inst.gaze_info - mu_g) / (sigma_g + 1e-8)
        inst.head_info = (inst.head_info - mu_h) / (sigma_h + 1e-8)
        inst.face_info = (inst.face_info - mu_f) / (sigma_f + 1e-8)

    # Salva dataset normalizzato
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(ds, f)
    print(f"[Saved] Normalized {split_name} -> {out_path}")


if __name__ == "__main__":
    # Imposta seed
    utils.seed_everything(utils.seed)

    # Configurazione preprocess
    cfg = Config_02_normalization

    for split in ["train", "validation", "test"]:
        normalize_dataset(split, cfg)

