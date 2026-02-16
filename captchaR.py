# %%
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F
#from gdown import download
from PIL import Image

# %%
# download(id='18IZn5DroVvTkGJEKn5w15RuT61ET9HP0', output='public-clean.png', quiet=False)
# download(id='1y3xX9VrM7EtYf1W-3vr-PETE19FnTKHR', output='public-clean.txt', quiet=False)

# download(id='1cvDlXQrkLvR_tQBydyms-dt-VSjB7RlG', output='public-noisy.png', quiet=False)
# download(id='1V-e76Q8Op3FWFEqb42bvJKllM5LfCdVS', output='public-noisy.txt', quiet=False)

# download(id='1JfbYOpBHNrlz-fqgOtLZ8QDidxznug7U', output='private-clean.png', quiet=False)
# download(id='1WtozDPV0FjmPthKBBiqBPfd-lhCMxfzk', output='private-noisy.png', quiet=False)

# %%
import os
def showImage(tensor):
    transform = transforms.ToPILImage()
    return transform(1-tensor)
def readExamples(file_name):
    transform = transforms.ToTensor()
    image = Image.open(file_name + '.png').convert('L')
    tensor = 1-transform(image)
    X = tensor.reshape(-1,1,28,140)
    if os.path.exists(file_name + '.txt'):
        with open(file_name + '.txt') as f:
            lines = f.readlines()
    else:
        lines = ['']*X.shape[0]
    return X, lines

examples, results = readExamples('public-clean') # or 'public-noisy' or 'private-clean' or 'private-noisy'
print("Example label line:", results[0].strip())
showImage(examples[0])

# %% [markdown]
# ## Λήψη του MNIST dataset
# 
# Θα κατεβάσουμε το κλασικό MNIST dataset μέσω του torchvision και θα το μετατρέψουμε σε tensors ώστε να χρησιμοποιηθεί αργότερα στην εκπαίδευση.
# 

# %%
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# %%
image, label = train_dataset[0]
print(label)
showImage(image)

# %% [markdown]
# # Η Βασική Ιδέα
# 
# Για να αυτοματοποιήσουμε την επίλυση των MNIST-CAPTCHA, πρέπει να εκπαιδεύσουμε ένα νευρωνικό που θα μπορεί να αποκωδικοποιήσει το μαθηματικό πρόβλημα από τα ψηφία του MNIST.
# 
# Το πρώτο βήμα είναι να εκπαιδεύσουμε ένα νευρωνικό δίκτυο που μπορεί να αναγνωρίσει να ψηφία ένα ένα.
# 
# Το επόμενο βήμα είναι να ξεχωρίσουμε τα ψηφία από το Captcha πρόβλημα, να χρησιμοποιήσουμε το εκπαιδευμένο νευρωνικό μας δίκτυο, και μετά να υπολογίσουμε την απάντηση, έχοντας βρεί τους σωστούς αριθμούς.

# %% [markdown]
# ## Απλό νευρωνικό δίκτυο
# 
# Ως ελάχιστη εκκίνηση, ορίζουμε ένα μοντέλο με ένα μόνο γραμμικό επίπεδο (layer) `nn.Linear(28*28, 10)` το οποίο ισοπεδώνει τα pixels και παράγει logits για τις 10 κλάσεις του MNIST.
# 

# %%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 320)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
torch.manual_seed(42)
model = CNN()
print(model)


# %% [markdown]
# ## Υλοποίηση εκπαίδευσης
# 
# Στο επόμενο βήμα θα χρειαστεί να γράψετε τη δική σας συνάρτηση εκπαίδευσης ώστε να προσαρμόσετε το μοντέλο στα δεδομένα και να το επεκτείνετε για τη λύση του MNIST CAPTCHA.
# 

# %%
# TODO: Γράψτε τον κώδικα εκπαίδευσης του μοντέλου
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

#Εκπαίδευση του μοντέλου
if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")
print("Using device:", device)
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64)

EPOCHS = 7

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc*100:.2f}%")


THRESHOLD = None

# %%
# Με βάση το εκπαιδευμένο μοντέλο υλοποιήστε μια συνάρτηση που υπολογίζει την πράξη στο δοσμένο παράδειγμα
def segment_captcha(x):
    # x shape: (1, 28, 140)
    return [x[:, :, i*28:(i+1)*28] for i in range(5)]

def extract_operators(levels=['public-clean', 'public-noisy']):
    ops, labels = [], []
    for level in levels:
        X, lines = readExamples(level)
        for img, line in zip(X, lines):
            op_img = img[:, :, 56:84]  # 28x28 operator block
            ops.append(op_img)

            if line.strip():
                labels.append("+" if "+" in line else "-")
            else:
                labels.append(None)
    return ops, labels

def compute_pixel_sums(ops, labels):
    plus_vals, minus_vals = [], []
    for img, label in zip(ops, labels):
        val = img.sum().item()
        if label == "+": plus_vals.append(val)
        if label == "-": minus_vals.append(val)
    return plus_vals, minus_vals

def find_operator_threshold():
    ops, labels = extract_operators()
    print("Counts: + =", labels.count("+"), ", - =", labels.count("-"))

    print("Total operators:", len(labels))
    print("Unique labels:", set(labels))
    print("First 20 labels:", labels[:20])

    plus_vals, minus_vals = compute_pixel_sums(ops, labels)

    mean_plus = np.mean(plus_vals)
    mean_minus = np.mean(minus_vals)
    threshold = (mean_plus + mean_minus) / 2

    print("Mean + :", mean_plus)
    print("Mean - :", mean_minus)
    print("Chosen threshold:", threshold)

    return threshold

# Compute threshold ONCE
THRESHOLD = find_operator_threshold()

def classify_operator(img):
    arr = img.squeeze().numpy()

    horizontal = arr[13:15, :].sum()   # middle horizontal stroke
    vertical   = arr[:, 13:15].sum()   # middle vertical stroke

    # plus has a strong vertical stroke
    if vertical > horizontal * 0.6:
        return "+"
    else:
        return "-"

# Predict results for a single captcha image
def predict(model, x):
    model.eval()

    # 1. Segment into 5 symbols
    A_img, B_img, op_img, C_img, D_img = segment_captcha(x)

    # 2. Predict digits
    with torch.no_grad():
        # A = model(A_img.to(device)).argmax().item()
        # B = model(B_img.to(device)).argmax().item()
        # C = model(C_img.to(device)).argmax().item()
        # D = model(D_img.to(device)).argmax().item()

        A = model(A_img.unsqueeze(0).to(device)).argmax(1).item()
        B = model(B_img.unsqueeze(0).to(device)).argmax(1).item()
        C = model(C_img.unsqueeze(0).to(device)).argmax(1).item()
        D = model(D_img.unsqueeze(0).to(device)).argmax(1).item()

    # 3. Predict operator
    op = classify_operator(op_img)

    # 4. Compute result
    left = 10*A + B
    right = 10*C + D
    result = left + right if op == '+' else left - right

    # 5. Format output
    print(f"Predicted: {A}{B}{op}{C}{D}={result}")
    return f"{A}{B}{op}{C}{D}={result}"


# %% [markdown]
# ## Αποθήκευση Απαντήσεων

# %%
import json
answers = {}
for level in ['public-clean', 'public-noisy', 'private-clean', 'private-noisy']:
    examples, _ = readExamples(level)
    answers[level] = '\n'.join([predict(model, example) for example in examples])
with open('answers.json', 'w') as f:
    json.dump(answers, f)



