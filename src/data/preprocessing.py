import polars as pl
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def create_sliding_windows(df, window_size=128, step_size=64):
    """
    Slices continuous sensor data into overlapping 3D tensors.
    """
    X_list = []
    y_list = []
    meta_list = []
    
    # Filter out unannotated 'null' rows
    clean_df = df.filter(
        pl.col("gt").is_not_null() & (pl.col("gt") != "null")
    )
    
    # Group by User, Device, and Activity to prevent boundary leakage
    groups = clean_df.group_by(["User", "Device", "gt"])
    
    for (user, device, activity), group_data in groups:
        sensors = group_data.select(["x", "y", "z"]).to_numpy()
        n_rows = len(sensors)
        
        if n_rows < window_size:
            continue
            
        for start in range(0, n_rows - window_size + 1, step_size):
            end = start + window_size
            window = sensors[start:end]
            
            X_list.append(window)
            y_list.append(activity)
            meta_list.append((user, device))
            
    return np.array(X_list), np.array(y_list), np.array(meta_list)


def extract_tabular_features(X_tensor):
    """
    Collapses a 3D tensor (Samples, Time, Channels) into a 2D tabular matrix
    by calculating statistical summaries over the time dimension.
    """
    means = np.mean(X_tensor, axis=1)
    stds = np.std(X_tensor, axis=1)
    maxs = np.max(X_tensor, axis=1)
    mins = np.min(X_tensor, axis=1)
    
    X_tabular = np.hstack([means, stds, maxs, mins])
    
    channels = ['x', 'y', 'z']
    stats = ['mean', 'std', 'max', 'min']
    col_names = [f"{ch}_{st}" for st in stats for ch in channels]
    
    return pd.DataFrame(X_tabular, columns=col_names)

if __name__ == "__main__":
    raw_data_path = "data/raw/Phones_accelerometer.csv"
    processed_dir = "data/processed"
    os.makedirs(processed_dir,exist_ok=True)
    save_path = os.path.join(processed_dir,'hhar_windows.npz')

    print("Loading raw data...")
    df = pl.read_csv(raw_data_path)

    print("Slicing continuous data into overlapping windows...")
    X_list,y_list,meta_list = create_sliding_windows(df)

    print("Encoding categorical labels...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_list)
    print(f"Classes mapped: {encoder.classes_}")

    print("Saving processed tensors to disk...")
    np.savez_compressed(
        save_path,
        X= X_list,
        y=y_encoded,
        meta_list = meta_list,
        classes = encoder.classes_
    )
    print(f"Success! Processed data saved to {save_path}")
    print(f"Total windows generated: {len(X_list)}")