# SIR_ML_MODEL: Learning Deterministic SIR Dynamics from Stochastic Simulations

This project investigates how machine learning can be used to recover the deterministic structure of the **SIR epidemiological model from stochastic simulations**. The SIR model is a foundational compartmental model in epidemiology that describes the spread of infectious diseases through three population states: Susceptible (S), Infectious (I), and Recovered (R). While the deterministic form of the model is expressed through a system of ordinary differential equations (ODEs), real-world epidemics often involve inherent randomness due to population fluctuations and transmission events.
To bridge this gap, the project generates data from stochastic simulations (e.g., using Gillespie’s algorithm) and trains neural networks to approximate the expected behavior of the underlying ODE system. This provides a **data-driven** approach to reconstructing dynamical systems, offering insights into how machine learning can complement traditional analytical modeling in the context of complex, noisy biological processes.

---

## Project Goals

- **Generate stochastic SIR simulation data**
- **Preprocess and normalize** the simulation data
- **Train an MLP (Multi-Layer Perceptron)** to learn the expected dynamics (S, I, R)
- **Visualize and evaluate** the learned predictions
- Set the stage for future **symbolic regression** or **autodiff-based** modeling

---

## Model Overview

The model learns a mapping:

(beta, gamma, time) → (S_mean, I_mean, R_mean)


Where:
- `beta` is the transmission rate
- `gamma` is the recovery rate
- `time` is the time index
- `S_mean`, `I_mean`, and `R_mean` are average population counts over stochastic simulations

---

## Technologies Used

- Python 
- PyTorch 
- NumPy & Pandas
- Matplotlib
- Scikit-learn

---

## Project Structure

```bash
SIR_ML_MODEL/
├── data/
│   └── processed/
│       └── sir_mean.csv       # Preprocessed mean (S, I, R) dataset
├── models/
│   └── sir_mlp.pt             # Trained PyTorch model
├── notebooks/
│   └── main.ipynb             # Full training + evaluation notebook
├── requirements.txt           # Python dependencies
└── README.md                  # You're here!
```
---

## Getting Started

- Clone the repo:
- git clone https://github.com/Awshae/SIR_ML_MODEL.git
- Install dependencies:
- pip install -r requirements.txt
- Run the notebook:
- Open notebooks/main.ipynb and run all cells to train or evaluate the SIRNet model.

---

## Results

 - High R² values for all three compartments (S, I, R)
 - Accurate prediction of epidemic curves for various β-γ combinations
 - Smooth learned dynamics with visual plots comparing true vs predicted values

---

## Future Work

 - Add support for symbolic regression (e.g., using PySR)
 - Incorporate neural differential equation layers (Neural ODEs)
 - Allow uncertainty quantification over stochastic simulations
 - Make the model interactive via a web app
 
 ---

## Contributing

Want to improve the model, add symbolic capabilities, or enhance the visualizations? Open a pull request or issue!

