import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import shutil

# Clear models/ directory
if os.path.exists("models"):
    shutil.rmtree("models")
os.makedirs("models", exist_ok=True)

# Load and preprocess data 
df = pd.read_csv("data/processed/sir_mean.csv")
X = df[["beta", "gamma", "time"]].values
y = df[["S", "I", "R"]].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save scalers for future inference
pd.DataFrame({'mean': scaler_X.mean_, 'scale': scaler_X.scale_}).to_csv("models/X_scaler.csv", index=False)
pd.DataFrame({'mean': scaler_y.mean_, 'scale': scaler_y.scale_}).to_csv("models/y_scaler.csv", index=False)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define Model
class SIRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

# Training
model = SIRNet()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

print("Training SIRNet model...")
for epoch in range(1000):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}")

# Save Model 
torch.save(model.state_dict(), "models/sir_mlp.pt")
print("Model saved to models/sir_mlp.pt")
