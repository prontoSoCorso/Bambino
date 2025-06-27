''' preprocessing_functions.py '''
import numpy as np
from scipy.signal import savgol_filter, medfilt, detrend


def _preprocess_channel(data: np.ndarray,
                        window_length: int = 11,
                        polyorder: int = 2,
                        median_kernel: int = 5) -> np.ndarray:
    """
    Applicazione di detrend, Savitzky-Golay smoothing e filtro mediano.

    Args:
        data: array 2D (timesteps x features)
    Returns:
        Processed array same shape.
    """
    # 1) Detrending per colonna
    data_dt = detrend(data, axis=0)

    # 2) Savitzky-Golay smoothing
    # window_length deve essere odd e <= num timesteps
    wl = min(window_length, data_dt.shape[0] // 2 * 2 - 1)
    if wl < polyorder + 2:
        wl = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    data_sg = savgol_filter(data_dt, window_length=wl, polyorder=polyorder, axis=0)

    # 3) Median filter lungo ogni feature
    # medfilt richiede kernel size odd scalar
    data_med = np.stack([medfilt(data_sg[:, i], kernel_size=median_kernel)
                         for i in range(data_sg.shape[1])], axis=1)

    return data_med

"""
Savitzky_Golay window
--> Deve essere odd e significativamente più piccolo di 250. 
--> Tipicamente si usa tra il 5% e il 15% della lunghezza della finestra, quindi tra 13 e 37 campioni.
--> prendo range 21-31
"""

def preprocess_gaze(gaze: np.ndarray) -> np.ndarray:
    """Preprocessing specifico per gaze_info"""
    return _preprocess_channel(gaze, window_length=21, polyorder=3, median_kernel=7)


def preprocess_head(head: np.ndarray) -> np.ndarray:
    """Preprocessing specifico per head_info"""
    return _preprocess_channel(head, window_length=25, polyorder=3, median_kernel=5)


def preprocess_face(face: np.ndarray) -> np.ndarray:
    """Preprocessing specifico per face_info"""
    return _preprocess_channel(face, window_length=31, polyorder=3, median_kernel=7)
