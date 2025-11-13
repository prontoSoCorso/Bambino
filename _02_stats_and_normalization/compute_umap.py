import os
import sys
import numpy as np
from pathlib import Path
from torch.utils.data import ConcatDataset

# Project path setup
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root.name != "Bambino":
    project_root = project_root.parent
sys.path.append(str(project_root))

# Imports
from config import settings
from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.BoaOpenFaceDataset import BoaOpenFaceDataset
from _utils_ import plot_utils



# Modalità: "all" per unire train+val+test, "single" per usarne uno solo
MODE = "single"             # "all" o "single"
SINGLE_DS = {"train", "validation", "test"}      # "train" / "validation" / "test", usato solo se MODE == "single"
INTERACTIVE = False       # True per Plotly/Dash, False per static Matplotlib
PORT = 8050               # Porta per il server Dash (se INTERACTIVE)
OUTPUT_DIR = None         # Directory di output per static UMAP; None = stessa cartella dello script
MODALITIES = ({'g': 'gaze_info',
               'f': 'face_info',
               'h': "head_info"
               })
ALL_COMB = True
DATA_TYPE = "normalized"


def load_datasets(keys, data_type):
    """Carica i BoaOpenFaceDataset indicati in keys e setta il .dataset_name."""
    datasets = []
    for key in keys:
        path  = settings.get_dataset_path(data_type, getattr(settings, f"{key}_filename"))
        ds = BoaOpenFaceDataset.load_dataset(path)
        ds.dataset_name = key
        datasets.append(ds)
    return datasets

def merge_with_concat(datasets):
    """
    Usa ConcatDataset per unire le istanze, poi aggiunge manualmente
    gli attributi .instances, .audio_groups e .trial_id_stats.
    """
    # 1) Concat
    concat_ds = ConcatDataset(datasets)

    # 2) Rigenero .instances come lista
    concat_ds.instances = []
    for ds in datasets:
        concat_ds.instances.extend(ds.instances)

    # 3) Metadati BOA
    concat_ds.audio_groups = list(
        np.unique([inst.audio for inst in concat_ds.instances])
    )
    trial_ids = [inst.trial_id for inst in concat_ds.instances]
    concat_ds.trial_id_stats = (np.mean(trial_ids), np.std(trial_ids))

    # 4) Mantengo anche un nome & output_dir coerenti
    concat_ds.dataset_name = "all"
    concat_ds.working_dir   = datasets[0].working_dir
    concat_ds.output_dir    = os.path.join(
        concat_ds.working_dir, "results", concat_ds.dataset_name
    )
    os.makedirs(concat_ds.output_dir, exist_ok=True)

    return concat_ds


def main(ds, mods, data_type):
    # Quali split caricare
    ds_keys = ["train", "validation", "test"] if MODE == "all" else [ds]
    datasets = load_datasets(ds_keys, data_type=data_type)

    # Merge o singolo
    if len(datasets) > 1:
        dataset = merge_with_concat(datasets)
    else:
        dataset = datasets[0]

    # Dove salvare l’UMAP statico
    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else os.path.join(current_file.parent, "umap_results", data_type)
    os.makedirs(out_dir, exist_ok=True)

    # Eseguo UMAP
    if INTERACTIVE:
        plot_utils.compute_UMAP_plotly(
            dataset,
            port=PORT
        )
    else:
        filename = "umap_" + ("_".join(mods.keys()) + "_") + ("all" if MODE == "all" else ds)
        plot_utils.compute_UMAP(
            dataset,
            output_dir=str(out_dir),
            filename=filename,
            modalities=mods
        )


if __name__ == "__main__":
    # e.g.: python run_umap.py --datasets train --output-dir umap_train
    # e.g.: python run_umap.py --datasets train test --interactive --port 8051
    from itertools import combinations

    for ds in SINGLE_DS:
        keys = list(MODALITIES.keys())
        if ALL_COMB:

            # looping over tutte le combinazioni di lunghezza 1 fino a len(keys)
            for r in range(1, len(keys) + 1):
                for combo in combinations(keys, r):
                    # combo è una tupla, es. ('f', 'g')
                    mods = {k: MODALITIES[k] for k in combo}
                    main(ds=ds, mods=mods, data_type=DATA_TYPE)

        else:
            main(ds=ds, mods=MODALITIES, data_type=DATA_TYPE)
