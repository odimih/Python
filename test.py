from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(2, 2)  # Input dimension is 2, output dimension is 2 (one for each class)

    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x,dim=1)
        return x

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    xmps = torch.ones(1, device=mps_device)
    print (xmps)
else:
    print ("MPS device not found.")

# Seed for reproducibility
np.random.seed(42)

# Generate positive examples in the upper right quadrant
X_positive = np.random.uniform(0.5,1.0,(20,2))

# Generate negative examples in the lower left quadrant
X_negative = np.random.uniform(0.0, 0.5,(20,2))

model = SimpleClassifier()
# Set weights and biases
model.fc.weight = nn.Parameter(torch.tensor([[-1.0, 1.0], [1.0, -1.0]]))
model.fc.bias = nn.Parameter(torch.tensor([-0.0, 0.0]))

# Select a few examples
points = torch.tensor([[0.1, 0.15],[0.8,0.8],[0.5,0.45]], dtype=torch.float32)

# Feed them through the network
outputs = model(points)

print("Outputs for selected points:\n", outputs)



X = np.vstack((X_positive,X_negative))
y = np.array([1]*20 + [0]*20)  # Correctly assigning labels to each class

# τα δεδομένα (X,y) τα έχουμε δημιουργήσει πιο πάνω
tensor_x_train = torch.tensor(X, dtype=torch.float32) # transform to torch tensor
tensor_y_train = torch.tensor(y, dtype=torch.long)     # labels should be torch.long for classification

m = 32 # Batch size
data_train = TensorDataset(tensor_x_train, tensor_y_train) # create your dataset
train_loader = DataLoader(data_train, batch_size=m, shuffle=True) # create your dataloader with training data

def train(model, train_loader, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.NLLLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    train_losses = []
    train_accuracies = []  # List to store accuracy for each epoch
    best_accuracy = 0  # Best accuracy found
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()  # Update the learning rate

        train_losses.append(total_loss / len(train_loader))

        model.eval()
        total_train_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                total_train_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        epoch_loss = total_train_loss / len(train_loader)
        epoch_accuracy = correct / total
        train_accuracies.append(epoch_accuracy)

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_losses[-1]}, Training Accuracy: {epoch_accuracy:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Loaded the best model from epoch {best_epoch} with Training Accuracy: {best_accuracy:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accuracies, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    return

torch.manual_seed(42)
model = SimpleClassifier()
model.to(mps_device)
optimizer = optim.Adam(model.parameters(), lr=0.1)
epochs = 2
torch.manual_seed(42)
train(model, train_loader,optimizer, epochs)

plot_decision_boundary(model.to('cpu'), X, y)
