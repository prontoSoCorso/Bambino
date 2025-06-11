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
from config import utils
from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.BoaOpenFaceDataset import BoaOpenFaceDataset
from _utils_ import plot_utils



# Modalità: "all" per unire train+val+test, "single" per usarne uno solo
MODE = "single"             # "all" o "single"
SINGLE_DS = "train"      # "train" / "validation" / "test", usato solo se MODE == "single"
INTERACTIVE = True       # True per Plotly/Dash, False per static Matplotlib
PORT = 8050               # Porta per il server Dash (se INTERACTIVE)
OUTPUT_DIR = None         # Directory di output per static UMAP; None = stessa cartella dello script


def load_datasets(keys):
    """Carica i BoaOpenFaceDataset indicati in keys e setta il .dataset_name."""
    datasets = []
    for key in keys:
        path = getattr(utils, f"{key}_path")
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


def main():
    # Quali split caricare
    ds_keys = ["train", "validation", "test"] if MODE == "all" else [SINGLE_DS]
    datasets = load_datasets(ds_keys)

    # Merge o singolo
    if len(datasets) > 1:
        dataset = merge_with_concat(datasets)
    else:
        dataset = datasets[0]

    # Dove salvare l’UMAP statico
    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else current_file.parent

    # Eseguo UMAP
    if INTERACTIVE:
        plot_utils.compute_UMAP_plotly(
            dataset,
            port=PORT
        )
    else:
        filename = "umap_" + ("all" if MODE == "all" else SINGLE_DS)
        plot_utils.compute_UMAP(
            dataset,
            output_dir=str(out_dir),
            filename=filename
        )


if __name__ == "__main__":
    # e.g.: python run_umap.py --datasets train --output-dir umap_train
    # e.g.: python run_umap.py --datasets train test --interactive --port 8051
    main()
