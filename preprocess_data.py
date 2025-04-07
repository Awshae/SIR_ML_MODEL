import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

# Directories
raw_dir = "data/raw"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# Load metadata
metadata = pd.read_csv("data/simulations_metadata.csv")
grouped = metadata.groupby(["beta", "gamma"])

# Fixed time grid
time_points = np.linspace(0, 100, 101)
all_data = []

print("Processing simulation data...")

for (beta, gamma), group in tqdm(grouped, desc="Beta-Gamma Combos"):
    S_list, I_list, R_list = [], [], []

    for fname in group["filename"]:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        # Standardize column names in case they are inconsistent
        df.rename(columns={
            "mean_S": "S", "S_mean": "S",
            "mean_I": "I", "I_mean": "I",
            "mean_R": "R", "R_mean": "R"
        }, inplace=True)

        if not all(col in df.columns for col in ["S", "I", "R", "time"]):
            continue

        df_interp = df.set_index("time").reindex(time_points).interpolate("index").fillna(method="ffill")
        S_list.append(df_interp["S"].values)
        I_list.append(df_interp["I"].values)
        R_list.append(df_interp["R"].values)

    if not S_list:
        continue

    mean_S = np.mean(S_list, axis=0)
    mean_I = np.mean(I_list, axis=0)
    mean_R = np.mean(R_list, axis=0)

    for i, t in enumerate(time_points):
        all_data.append({
            "beta": beta,
            "gamma": gamma,
            "time": t,
            "S": mean_S[i],
            "I": mean_I[i],
            "R": mean_R[i],
            "num_simulations": len(S_list),
            "timestamp": datetime.now().isoformat()
        })

# Save final processed dataset
processed_df = pd.DataFrame(all_data)
processed_df.to_csv(os.path.join(output_dir, "sir_mean.csv"), index=False)

print(f"Saved processed data to {output_dir}/sir_mean.csv with shape {processed_df.shape}")
