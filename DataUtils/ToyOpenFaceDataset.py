# Import packages
import random
import numpy as np
import copy

from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.OpenFaceInstance import OpenFaceInstance

# Class
class ToyOpenFaceDataset(OpenFaceDataset):
    # Define class attributes
    sex_groups = ["Female", "Male"]
    time_threshold = 4.5

    # Define folders
    data_fold = "data/toy/"
    results_fold = "results/toy/"

    def __init__(self, dataset_name, working_dir, file_name, data_instances=None, is_boa=False):
        super().__init__(dataset_name, working_dir, file_name, data_instances, is_boa, is_toy=True)

    def __getitem__(self, idx):
        x, y, extra = super().__getitem__(idx)
        age, trial_id_categorical, trial_id = extra

        instance = self.instances[idx]
        sex = OpenFaceDataset.preprocess_label(instance.sex)

        return x, y, [age, trial_id_categorical, trial_id, sex]

    def show_durations(self, get_parameters=False):
        size_samples = [instance.face_info.shape[0] for instance in self.instances if instance.trial_type and
                        instance.clinician_pred]
        true_trial_durations = [sample / OpenFaceDataset.FRAME_RATE for sample in size_samples]
        print("Correctly classified stimulus trials: mean sequence duration =", str(np.mean(true_trial_durations)) + "s",
              "(std = " + str(np.std(true_trial_durations)) + ", n = " + str(len(true_trial_durations)) + ")")

        false_trial_durations = [instance.face_info.shape[0] / OpenFaceDataset.FRAME_RATE for instance in self.instances
                                 if instance.trial_type and not instance.clinician_pred]
        print("Wrongly classified stimulus trials: mean sequence duration =", str(np.mean(false_trial_durations)) + "s",
              "(std = " + str(np.std(false_trial_durations)) + ", n = " + str(len(false_trial_durations)) + ")")

        control_durations = [instance.face_info.shape[0] / OpenFaceDataset.FRAME_RATE for instance in self.instances
                             if not instance.trial_type]
        print("Control trials: mean sequence duration =", str(np.mean(control_durations)) + "s", "(std = " +
              str(np.std(control_durations)) + ", n = " + str(len(control_durations)) + ")")

        if get_parameters:
            return size_samples

    def compute_statistics(self, trial_id_stats=None, return_output=False):
        ages_categorical, ages_categorical_all, trials_categorical = super().compute_statistics(trial_id_stats,
                                                                                                True)

        # Count gender (at patient level)
        sexes = []
        for pt_id in self.ids:
            for instance in self.instances:
                if instance.pt_id == pt_id:
                    sex = instance.sex
                    sexes.append(sex)
                    break
        OpenFaceDataset.draw_hist(sexes, 2, "Gender distribution", self.preliminary_dir + "sex_distr",
                                  self.sex_groups)

        # Count instances by both age and gender (at patient level)
        OpenFaceDataset.interaction_count(ages_categorical, sexes, self.AGE_GROUPS, self.sex_groups,
                                          "Age (categorical)", "Gender",
                                          self.preliminary_dir + "age_vs_gender.png")

        # Count instances by both gender and number of trials
        sexes_all = [instance.sex for instance in self.instances]
        OpenFaceDataset.interaction_count(sexes_all, trials_categorical, self.sex_groups, self.TRIAL_ID_GROUPS,
                                          "Gender", "Trial ID (categorical)",
                                          self.preliminary_dir + "sex_vs_trial.png")

        if return_output:
            return ages_categorical, ages_categorical_all, trials_categorical, sexes, sexes_all
