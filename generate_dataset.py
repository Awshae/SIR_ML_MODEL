from simulate_sir import gillespie_SIR
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import datetime

# Create output directory
raw_dir = "data/raw"
os.makedirs(raw_dir, exist_ok=True)

# Define parameter grid
beta_vals = np.linspace(0.1, 1.0, 10)
gamma_vals = np.linspace(0.1, 1.0, 10)
num_simulations = 50

# Metadata storage
metadata = []
timestamp = datetime.datetime.now().isoformat()

print("Generating SIR simulation data...")

# Run simulations
for beta in tqdm(beta_vals, desc="Beta Loop"):
    for gamma in gamma_vals:
        for sim_id in range(num_simulations):
            df = gillespie_SIR(S0=95, I0=5, R0=0, beta=beta, gamma=gamma, max_time=100)

            # Rename columns if necessary
            rename_map = {
                "S_mean": "S", "mean_S": "S",
                "I_mean": "I", "mean_I": "I",
                "R_mean": "R", "mean_R": "R"
            }
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

            # Optional check
            if not all(col in df.columns for col in ["S", "I", "R", "time"]):
                print(f"Warning: Missing expected columns in simulation {sim_id} for beta={beta:.2f}, gamma={gamma:.2f}")

            # Save to CSV
            filename = f"b{beta:.2f}_g{gamma:.2f}_sim{sim_id}.csv"
            filepath = os.path.join(raw_dir, filename)
            df.to_csv(filepath, index=False)

            # Add to metadata
            metadata.append({
                "filename": filename,
                "beta": beta,
                "gamma": gamma,
                "sim_id": sim_id,
                "num_rows": len(df),
                "max_time": df['time'].max(),
                "generated_at": timestamp
            })

# Save metadata
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv("data/simulations_metadata.csv", index=False)

print(f"Generated {len(metadata)} simulations. Metadata saved to data/simulations_metadata.csv")
