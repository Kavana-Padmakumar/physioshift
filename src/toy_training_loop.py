import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. Custom Dataset Class
# =====================================================================
class SignalDataset(Dataset):
    """
    Simulates loading a harmonized signal file.
    Must implement __len__ and __getitem__.
    """
    def __init__(self, num_samples=100, signal_len=32):
        # Generate 100 random signal samples of size 32
        self.data = torch.randn(num_samples, signal_len)
        # Generate dummy targets based on input values
        self.targets = self.data.sum(dim=1, keepdim=True) + torch.randn(num_samples, 1) * 0.1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# =====================================================================
# 2. Dummy 2-Layer Neural Network
# =====================================================================
class DummyNetwork(nn.Module):
    """
    Simple 2-layer network architecture.
    """
    def __init__(self, input_dim=32, hidden_dim=16, output_dim=1):
        super(DummyNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# =====================================================================
# 3. 5-Epoch Training Loop Execution
# =====================================================================
def main():
    # Instantiate Data Loader & Model
    dataset = SignalDataset(num_samples=100, signal_len=32)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = DummyNetwork(input_dim=32, hidden_dim=16, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 5
    print("Executing 5-epoch PyTorch training mechanics...\n" + "-" * 40)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        for batch_x, batch_y in dataloader:
            # Step 1: Clear old gradients
            optimizer.zero_grad()

            # Step 2: Forward pass
            outputs = model(batch_x)

            # Step 3: Loss calculation
            loss = criterion(outputs, batch_y)

            # Step 4: Backward pass (Compute gradients)
            loss.backward()

            # Step 5: Optimizer step (Update weights)
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f}")

    print("-" * 40 + "\nExecution complete.")

if __name__ == "__main__":
    main()