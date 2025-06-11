# Import packages
import os, sys
import numpy as np

# Configurazione dei percorsi
parent_dir = os.path.dirname(os.path.abspath(__file__))
while not os.path.basename(parent_dir) == "Bambino":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import utils
from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.OpenFaceInstance import OpenFaceInstance

class BoaOpenFaceDataset(OpenFaceDataset):
    """
    BOA‐specific extension: adds sex, audio and speaker as extra labels,
    and BOA‐specific grouping for age and speaker side.
    """
    # Define class attributes
    SEX_GROUPS     = ["Female", "Male"]
    AGE_GROUPS_BOA = ["[3-5.5)", "[5.5-7]"]
    SPEAKER_GROUPS = ["Left", "Right"]

    def __init__(self,
                 dataset_name: str,
                 working_dir: str,
                 file_name: str,
                 data_instances: list = None,
                 audio_groups: list = None,
                 modalities: list = None):
        super().__init__(dataset_name, working_dir, file_name, data_instances, modalities)

        # override BOA‐specific age groups for stats()
        self.AGE_GROUPS = BoaOpenFaceDataset.AGE_GROUPS_BOA

        # set audio labels
        if audio_groups is not None:
            self.audio_groups = list(audio_groups)
        else:
            # np.unique returns ndarray, so wrap in list()
            self.audio_groups = list(np.unique([inst.audio for inst in self.instances]))


    def __getitem__(self, idx):
        x, y, extras = super().__getitem__(idx)
        age_t, trial_t, trial_id_t = extras

        inst        = self.instances[idx]
        sex_t       = OpenFaceDataset._to_label_tensor(inst.sex)
        audio_idx   = list(self.audio_groups).index(inst.audio)
        audio_t     = OpenFaceDataset._to_label_tensor(audio_idx)
        spk         = inst.speaker if inst.speaker is not None else -1
        speaker_t   = OpenFaceDataset._to_label_tensor(spk)

        return x, y, [age_t, trial_t, trial_id_t, sex_t, audio_t, speaker_t]
    
    def load_dataset(path: str, audio_groups: list = None, modalities: list = None):
        """
        Load a pickled BoaOpenFaceDataset and restore new attributes.
        """
        ds = OpenFaceDataset.load_dataset(path, modalities=modalities)
        # restore audio_groups if provided
        if audio_groups is not None:
            ds.audio_groups = list(audio_groups)
        else:
            ds.audio_groups = list(np.unique([inst.audio for inst in ds.instances]))
        return ds
    
    def compute_statistics(self, recalc_stats: bool = False):
        # set trial_id_stats if needed
        if self.trial_id_stats is None or recalc_stats:
            trials = [inst.trial_id for inst in self.instances]
            self.trial_id_stats = (np.mean(trials), np.std(trials))

        # 1) sex distribution
        sexes = [
            next(i for i in self.instances if i.pt_id == pid).sex
            for pid in self.ids
            ]
        self._draw_hist(sexes,
                        max_value=2,
                        labels=BoaOpenFaceDataset.SEX_GROUPS,
                        title="Sex Distribution",
                        out=os.path.join(self.output_dir, "sex_distribution"))

        # 2) age vs sex
        ages_cat = [
            OpenFaceInstance.categorize_age(
                next(i for i in self.instances if i.pt_id == pid).age
            )
            for pid in self.ids
            ]
        self._interaction_plot(ages_cat,
                               sexes,
                               self.AGE_GROUPS_BOA,
                               BoaOpenFaceDataset.SEX_GROUPS,
                               "Age vs Sex",
                               os.path.join(self.output_dir, "age_vs_sex"))

        # 3) sex vs trial count
        trial_cat_all = [
            OpenFaceInstance.categorize_trial_id(inst.trial_id, self.trial_id_stats)
            for inst in self.instances
        ]
        self._interaction_plot(sexes,
                               trial_cat_all,
                               BoaOpenFaceDataset.SEX_GROUPS,
                               OpenFaceDataset.TRIAL_ID_GROUPS,
                               "Sex vs Trial Count",
                               os.path.join(self.output_dir, "sex_vs_trial"))

        # 4) speaker side
        speakers = [
            inst.speaker if inst.speaker is not None else -1
            for inst in self.instances
        ]
        self._draw_hist(speakers,
                        max_value=2,
                        labels=BoaOpenFaceDataset.SPEAKER_GROUPS + ["Unknown"],
                        title="Speaker Side",
                        out=os.path.join(self.output_dir, "speaker_distribution"))

        # 5) audio groups
        audio_idxs = [list(self.audio_groups).index(inst.audio) for inst in self.instances]
        self._draw_hist(audio_idxs,
                        max_value=len(self.audio_groups),
                        labels=self.audio_groups,
                        title="Audio Groups",
                        out=os.path.join(self.output_dir, "audio_distribution"))


# Main
if __name__ == "__main__":
    utils.seed_everything(utils.seed)
    test_set = OpenFaceDataset.load_dataset(utils.validation_path)
    test_set.compute_statistics()
    print("Done BOA statistics.")

    print("sjhfweuihguiowehgl")