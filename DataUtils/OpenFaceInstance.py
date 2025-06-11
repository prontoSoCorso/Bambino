# Import packages
import numpy as np
import pandas as pd


# Class
class OpenFaceInstance:
    """
    Wraps a single participant-trial’s worth of OpenFace output (gaze, head, face).
    """

    # Define class attributes
    dim_dict = {"g": 8, "h": 13, "f": 17}
    dim_names = {"g": "Gaze direction", "h": "Head pose", "f": "Facial expression"}
    dim_labels = {"g": ["gaze_0_x", "gaze_1_x", "gaze_angle_x", "gaze_0_y", "gaze_1_y", "gaze_angle_y", "gaze_0_z",
                        "gaze_1_z"],
                  "h": ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Ry_smooth", "pose_Rz", "p_scale",
                        "p_rx", "p_ry", "p_rz", "p_tx", "p_ty"],
                  "f": ["Inner Brow Raiser", "Outer Brow Raiser", "Brow Lowerer", "Upper Lid Raiser", "Cheek Raiser",
                        "Lid Tightener", "Nose Wrinkler", "Upper Lip Raiser", "Lip Corner Puller", "Dimpler",
                        "Lip Corner Depressor", "Chin Raiser", "Lip stretcher", "Lip Tightener", "Lips part",
                        "Jaw Drop", "Blink"]}
    subplot_settings = {"g": [15, 10, 2, 4], "h": [15, 10, 3, 5], "f": [15, 10, 4, 5]}

    def __init__(self, trial_data: pd.DataFrame):
        arr = trial_data.to_numpy()
        # column indices (you may want to double-check these against your CSV!)
        pt_ind, sex_ind, age_ind, trial_ind, trl_type_ind = 0, 1, 3, 4, 5
        # audio & speaker
        raw_audio = arr[0, 6]
        self.audio = raw_audio[:-4].replace("_", " ")
        raw_spk = arr[0, 7].lower()
        self.speaker = 0 if raw_spk == "left" else (1 if raw_spk == "right" else None)
        # fixed fields
        self.pt_id      = arr[0, pt_ind]
        self.sex        = int(arr[0, sex_ind])
        self.age        = float(arr[0, age_ind])
        self.trial_id   = float(arr[0, trial_ind])
        self.trial_type = int(arr[0, trl_type_ind])
        # time‐series fields (with linear interp. of any NaNs)
        gaze_cols = range(17, 25)
        head_cols = range(25, 38)
        face_cols = range(38, 55)

        # Interpolation
        self.gaze_info = np.stack([pd.Series(arr[:, i].astype(np.float32)).interpolate() for i in gaze_cols], axis=1)
        self.head_info = np.stack([pd.Series(arr[:, i].astype(np.float32)).interpolate() for i in head_cols], axis=1)
        self.face_info = np.stack([pd.Series(arr[:, i].astype(np.float32)).interpolate() for i in face_cols], axis=1)

        # after interpolation, back- and forward-fill any edge NaNs then zero any that survive
        self.gaze_info = pd.DataFrame(self.gaze_info).fillna(method='bfill').fillna(method='ffill').to_numpy()
        self.head_info = pd.DataFrame(self.head_info).fillna(method='bfill').fillna(method='ffill').to_numpy()
        self.face_info = pd.DataFrame(self.face_info).fillna(method='bfill').fillna(method='ffill').to_numpy()

        # finally, guard against any remaining NaN/infs
        self.gaze_info = np.nan_to_num(self.gaze_info, nan=0.0, posinf=0.0, neginf=0.0)
        self.head_info = np.nan_to_num(self.head_info, nan=0.0, posinf=0.0, neginf=0.0)
        self.face_info = np.nan_to_num(self.face_info, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def categorize_age(age: float) -> int:
        if  3.0 <= age <  5.5: return 0
        if  5.5 <= age <  7.5: return 1
        return None

    @staticmethod
    def categorize_trial_id(trial_id: float, stats: tuple) -> int:
        mu, sigma = stats
        b1, b2 = mu - 0.5*sigma, mu + 0.5*sigma
        if trial_id <  b1: return 0
        if b1 <= trial_id < b2: return 1
        return 2
