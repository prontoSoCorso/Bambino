# Import packages
import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

# Configurazione dei percorsi
parent_dir = os.path.dirname(os.path.abspath(__file__))
while not os.path.basename(parent_dir) == "Bambino":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import utils
from DataUtils.OpenFaceInstance import OpenFaceInstance

# Class
class OpenFaceDataset(Dataset):
    # Define class attributes
    TRIAL_TYPES = ["control", "stimulus"]
    AGE_GROUPS = ["7-11", "12-18", "19-24"]
    TRIAL_ID_GROUPS = ["0-30th percentiles", "30-70th percentiles", "70-100th percentiles"]
    FRAME_RATE = 25
    MIN_SEQUENCE_LENGTH = 300

    def __init__(self, 
                 dataset_name: str, 
                 working_dir: str, 
                 file_name: str, 
                 data_instances: list=None,
                 modalities: list = None):
        """
        Args:
            dataset_name: name to tag stored .pt and plots
            working_dir: directory where CSV lives and where .pt will be saved
            file_name: name of CSV file to read (with extension)
            data_instances: optional list of pre-built OpenFaceInstance
        """
        super().__init__()
        # seed / reproducibility
        utils.seed_everything(utils.seed)
        
        # paths
        self.dataset_name = dataset_name
        self.working_dir = working_dir
        self.data_path = os.path.join(working_dir, file_name)
        self.output_dir = os.path.join(working_dir, "results", dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # load raw CSV
        # store selected modalities (defaults to all)
        self.modalities = modalities if modalities is not None else ["g", "h", "f"]
        df = pd.read_csv(self.data_path)
        # normalize labels
        df["sex"] = df["sex"].map({"Boy": 1, "Girl": 0})
        df["trial_type"] = df["trial_type"].map({
            OpenFaceDataset.TRIAL_TYPES[0]: 0,
            OpenFaceDataset.TRIAL_TYPES[1]: 1
        })

        # build instances
        if data_instances is None:
            self.instances = []
            self.ids = df["participant_id"].unique()
            self.trials = df["trial_id"].unique()
            for pid in self.ids:
                for tid in self.trials:
                    sub = df[(df["participant_id"] == pid) & (df["trial_id"] == tid)]
                    if (sub.empty 
                        or sub["low_confidence_for_trial"].iat[0] 
                        or (sub["confidence"].iloc[:52] <= 0.5).all()):
                        continue
                    self.instances.append(OpenFaceInstance(sub))
        else:
            self.instances = data_instances
            self.ids = np.unique([inst.pt_id for inst in data_instances])

        self.trial_id_stats = None  # to be computed in compute_statistics()

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        # build full feature dict, then filter by selected modalities
        full_x = {
            "g": torch.tensor(inst.gaze_info, dtype=torch.float),
            "h": torch.tensor(inst.head_info, dtype=torch.float),
            "f": torch.tensor(inst.face_info, dtype=torch.float),
        }
        x = {k: full_x[k] for k in self.modalities}
        y = OpenFaceDataset._to_label_tensor(inst.trial_type)

        # extra metadata
        age_cat    = OpenFaceInstance.categorize_age(inst.age)
        trial_cat  = OpenFaceInstance.categorize_trial_id(inst.trial_id, self.trial_id_stats)

        extras = [
            OpenFaceDataset._to_label_tensor(age_cat),
            OpenFaceDataset._to_label_tensor(trial_cat),
            OpenFaceDataset._to_label_tensor(inst.trial_id),
        ]
        return x, y, extras

    @staticmethod
    def _to_label_tensor(val: int) -> torch.LongTensor:
        return torch.tensor([val], dtype=torch.long)
    
    @staticmethod
    def load_dataset(path: str, modalities: list = None):
        """
        Unpickle a saved dataset and back-fill any missing attributes
        (e.g. output_dir) introduced since it was pickled.
        """
        with open(path, "rb") as f:
            ds = pickle.load(f)

        # Back-fill output_dir if missing
        if not hasattr(ds, "output_dir"):
            ds.output_dir = os.path.join(ds.working_dir, "results", ds.dataset_name)
            os.makedirs(ds.output_dir, exist_ok=True)

        # Back-fill trial_id_stats if missing
        if not hasattr(ds, "trial_id_stats"):
            ds.trial_id_stats = None

        if modalities is not None:
            ds.modalities = modalities

        return ds
    
    def compute_statistics(self, recalc_stats: bool = False):
        """
        Produce and save:
          - trial type pie/bar plot
          - age histogram (raw + categorical)
          - trial count histogram (raw + categorical)
          - age vs. trial_id heatmap
          - sequence duration summary + list of < MIN_SEQUENCE_LENGTH
        """
        # set trial_id_stats if needed
        if self.trial_id_stats is None or recalc_stats:
            trials = [inst.trial_id for inst in self.instances]
            self.trial_id_stats = (np.mean(trials), np.std(trials))

        # 1) trial types
        n_stim   = sum(inst.trial_type == 1 for inst in self.instances)
        n_control = len(self.instances) - n_stim
        self._draw_pie_bar([n_control, n_stim],
                           labels=OpenFaceDataset.TRIAL_TYPES,
                           title="Trial Types",
                           out=os.path.join(self.output_dir, "trial_types"))

        # 2) ages
        ages = []
        ages_cat = []
        for pid in self.ids:
            inst = next(filter(lambda i: i.pt_id == pid, self.instances))
            ages.append(inst.age)
            ages_cat.append(OpenFaceInstance.categorize_age(inst.age))
        self._draw_hist(ages, max_value=OpenFaceDataset.AGE_GROUPS[-1], 
                        title="Age (months)", 
                        out=os.path.join(self.output_dir, "age_months"))
        self._draw_hist(ages_cat, max_value=len(OpenFaceDataset.AGE_GROUPS),
                        labels=OpenFaceDataset.AGE_GROUPS,
                        title="Age Categories",
                        out=os.path.join(self.output_dir, "age_categorical"))

        # 3) trial distribution
        trials = [inst.trial_id for inst in self.instances]
        self._draw_hist(trials, max_value=int(max(trials)),
                        title="Trial ID (raw)",
                        out=os.path.join(self.output_dir, "trial_raw"))
        trials_cat = [OpenFaceInstance.categorize_trial_id(t, self.trial_id_stats) for t in trials]
        self._draw_hist(trials_cat, max_value=len(OpenFaceDataset.TRIAL_ID_GROUPS),
                        labels=OpenFaceDataset.TRIAL_ID_GROUPS,
                        title="Trial ID (categorical)",
                        out=os.path.join(self.output_dir, "trial_categorical"))

        # 4) age vs. trial heatmap
        self._interaction_plot(ages_cat, trials_cat,
                               row_labels=OpenFaceDataset.AGE_GROUPS,
                               col_labels=OpenFaceDataset.TRIAL_ID_GROUPS,
                               title="Age vs Trial",
                               out=os.path.join(self.output_dir, "age_vs_trial"))

        # 5) durations
        durations = [inst.face_info.shape[0] / OpenFaceDataset.FRAME_RATE for inst in self.instances]
        print(f"Mean duration: {np.mean(durations):.2f}s (std {np.std(durations):.2f}s)")
        short = [(inst.pt_id, inst.trial_id, inst.face_info.shape[0]) 
                 for inst in self.instances if inst.face_info.shape[0] < OpenFaceDataset.MIN_SEQUENCE_LENGTH]
        for pid, tid, length in short:
            print(f"  → Patient {pid}, trial {tid}: only {length} frames")

    def _save(self, name: str):
        path = os.path.join(self.working_dir, f"{name}.pt")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Saved] {path}")
    
    @staticmethod
    def _draw_pie_bar(counts, labels, title, out, do_bar=False):
        plt.figure()
        if do_bar:
            plt.bar(labels, counts)
        else:
            plt.pie(counts, labels=labels, autopct="%1.1f%%")
        plt.title(title)
        ext = ".png" if do_bar else ".jpg"
        plt.savefig(out + ext, dpi=300)
        plt.close()

    @staticmethod
    def _draw_hist(data, max_value, title, out, labels=None):
        plt.figure()
        bins = len(labels) if labels is not None else min(50, int(max_value) + 1)
        plt.hist(data, bins=bins)
        if labels is not None:
            plt.xticks(range(len(labels)), labels, rotation=45)
        plt.title(title)
        plt.savefig(out + ".png", dpi=300)
        plt.close()

    @staticmethod
    def _interaction_plot(var1, var2, row_labels, col_labels, title, out):
        cm = np.zeros((len(row_labels), len(col_labels)), dtype=int)
        for v1, v2 in zip(var1, var2):
            cm[v1, v2] += 1
        plt.figure(figsize=(6, 4))
        plt.imshow(cm, cmap="Reds")
        plt.xticks(range(len(col_labels)), col_labels, rotation=45)
        plt.yticks(range(len(row_labels)), row_labels)
        plt.colorbar(label="Count")
        plt.title(title)
        plt.savefig(out + ".png", dpi=300, bbox_inches="tight")
        plt.close()