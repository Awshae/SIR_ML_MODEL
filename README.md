# 🧠 SIR_ML_MODEL: Learning Deterministic SIR Dynamics from Stochastic Simulations

This project uses machine learning to approximate the deterministic behavior of the **SIR epidemiological model** based on **stochastic simulations**. It serves as a GSoC-ready demonstration of modeling differential equation systems using neural networks.

---

## 📌 Project Goals

- 📈 **Generate stochastic SIR simulation data**
- 🧹 **Preprocess and normalize** the simulation data
- 🧠 **Train an MLP (Multi-Layer Perceptron)** to learn the expected dynamics (S, I, R)
- 📊 **Visualize and evaluate** the learned predictions
- 🔬 Set the stage for future **symbolic regression** or **autodiff-based** modeling

---

## 🧪 Model Overview

The model learns a mapping:

(beta, gamma, time) → (S_mean, I_mean, R_mean)


Where:
- `beta` is the transmission rate
- `gamma` is the recovery rate
- `time` is the time index
- `S_mean`, `I_mean`, and `R_mean` are average population counts over stochastic simulations

---

## 🧰 Technologies Used

- Python 🐍
- PyTorch ⚙️
- NumPy & Pandas
- Matplotlib
- Scikit-learn

---

## 📁 Project Structure

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

---

🚀 Getting Started

Clone the repo:
git clone https://github.com/Awshae/SIR_ML_MODEL.git
Install dependencies:
pip install -r requirements.txt
Run the notebook:
Open notebooks/main.ipynb and run all cells to train or evaluate the SIRNet model.

---

📊 Results

✅ High R² values for all three compartments (S, I, R)
📈 Accurate prediction of epidemic curves for various β-γ combinations
🔍 Smooth learned dynamics with visual plots comparing true vs predicted values
<p align="center"> <img src="assets/sample_plot.png" width="500"/> </p>