from pathlib import Path
import h5py
import numpy as np


def create_dataset(path : Path, save_dir : Path, save_name : str):
    """
    Extracts TIR1 channel and stores it in the save dir, no preprocessing
    Note : Spatial Resolution unchanged, kept at ~ 4km 
    """
    assert path.exists(), "Invalid path entered"
    with h5py.File(path,'r') as f:
        tir1 = f['IMG_TIR1'][:]
        tir1 = tir1.squeeze()
        
        calibrator = f['IMG_TIR1_TEMP'][:]
        tir1_calibrated = calibrator[tir1]
        
        
        save_dir.mkdir(parents = True,exist_ok = True) 
        
        save_path = save_dir / f"{save_name}.npy"
        np.save(save_path, tir1_calibrated)

def basic_preprocessor(path_to_h5, save_dir, save_name, factor=4):
    """
    Extracts the TIR1 channel, calibrates, downsamples to 10km, and saves as .npy
    """
    path_to_h5 = Path(path_to_h5)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    assert path_to_h5.exists(), "Invalid path given"

    with h5py.File(path_to_h5, 'r') as f:  # SAFELY OPENED
        channel_TIR1 = f['IMG_TIR1'][:]
        channel_TIR1_temp = f['IMG_TIR1_TEMP'][:]

    # Clip values to avoid indexing errors
    channel_TIR1 = np.clip(channel_TIR1, 0, len(channel_TIR1_temp) - 1)
    channel_TIR1_cal = channel_TIR1_temp[channel_TIR1.squeeze()]

    # Downsampling
    H, W = channel_TIR1_cal.shape
    new_h = (H // factor) * factor
    new_w = (W // factor) * factor
    cropped = channel_TIR1_cal[:new_h, :new_w]
    downsampled = cropped.reshape(new_h // factor, factor, new_w // factor, factor).mean(axis=(1, 3))

    save_path = save_dir / f"{save_name}.npy"
    np.save(save_path, downsampled)
