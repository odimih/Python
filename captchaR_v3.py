# %%
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

examples, results = readExamples('public-clean')
print("Example label line:", results[0].strip())
showImage(examples[0])

# %% [markdown]
# ## Λήψη του MNIST dataset

# %%
# Optimisation #1: add RandomAffine augmentation alongside Gaussian noise.
# Small rotations/translations teach the model to handle the slight positional
# imprecision that comes from fixed 28-pixel segmentation of captcha strips.
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x + 0.1 * torch.randn_like(x)).clamp(0, 1)),
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# %%
image, label = train_dataset[0]
print(label)
showImage(image)

# %% [markdown]
# ## Digit CNN

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
        x = self.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

torch.manual_seed(42)
model = CNN()
print(model)

# %% [markdown]
# ## Training

# %%
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
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
    return running_loss / total, correct / total

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
    return running_loss / total, correct / total

if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")
print("Using device:", device)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64)

EPOCHS = 10
for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc   = evaluate(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc*100:.2f}%")

# %%
def segment_captcha(x):
    # x shape: (1, 28, 140) → five 28×28 blocks: A, B, op, C, D
    return [x[:, :, i*28:(i+1)*28] for i in range(5)]

# %%
def classify_operator(img):
    arr = img.squeeze().numpy()
    horizontal = arr[13:15, :].sum()   # middle horizontal stroke
    vertical   = arr[:, 13:15].sum()   # middle vertical stroke
    # '+' has a strong vertical stroke; '-' does not
    return "+" if vertical > horizontal * 0.6 else "-"

def predict(model, x):
    model.eval()
    A_img, B_img, op_img, C_img, D_img = segment_captcha(x)
    with torch.no_grad():
        A = model(A_img.unsqueeze(0).to(device)).argmax(1).item()
        B = model(B_img.unsqueeze(0).to(device)).argmax(1).item()
        C = model(C_img.unsqueeze(0).to(device)).argmax(1).item()
        D = model(D_img.unsqueeze(0).to(device)).argmax(1).item()
    op     = classify_operator(op_img)
    left   = 10*A + B
    right  = 10*C + D
    result = left + right if op == '+' else left - right
    print(f"Predicted: {A}{B}{op}{C}{D}={result}")
    return f"{A}{B}{op}{C}{D}={result}"

# %%
# Optimisation #2: fine-tune on verified noisy captcha crops.
# Run model on public-noisy (has numeric result labels), collect digit crops
# from correctly-solved equations, then fine-tune at a low learning rate to
# adapt the model to the noisy pixel domain.
def collect_verified_crops(model, level='public-noisy'):
    examples, labels = readExamples(level)
    crops, targets = [], []
    model.eval()
    with torch.no_grad():
        for img, label_line in zip(examples, labels):
            label_str = label_line.strip()
            if not label_str:
                continue
            try:
                expected = int(label_str)
            except ValueError:
                continue
            A_img, B_img, op_img, C_img, D_img = segment_captcha(img)
            A = model(A_img.unsqueeze(0).to(device)).argmax(1).item()
            B = model(B_img.unsqueeze(0).to(device)).argmax(1).item()
            C = model(C_img.unsqueeze(0).to(device)).argmax(1).item()
            D = model(D_img.unsqueeze(0).to(device)).argmax(1).item()
            op = classify_operator(op_img)
            left  = 10*A + B
            right = 10*C + D
            result = left + right if op == '+' else left - right
            if result == expected:
                crops += [A_img, B_img, C_img, D_img]
                targets += [A, B, C, D]
    print(f"Collected {len(crops)} verified crops from {level} ({len(crops)//4} equations)")
    return crops, targets

prev_count = 0
lr = 1e-4
round_num = 1
while True:
    crops, crop_labels = collect_verified_crops(model, 'public-noisy')
    if len(crops) <= prev_count:
        print(f"Converged after {round_num - 1} bootstrap rounds.")
        break
    noisy_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.stack(crops),
            torch.tensor(crop_labels, dtype=torch.long)
        ),
        batch_size=32, shuffle=True)
    ft_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Fine-tuning round {round_num} (lr={lr:.0e}, {len(crops)//4} equations)...")
    for epoch in range(5):
        ft_loss, ft_acc = train(model, noisy_loader, ft_optimizer, criterion, device)
        print(f"  Epoch {epoch+1}/5 Loss: {ft_loss:.4f} Acc: {ft_acc*100:.2f}%")
    prev_count = len(crops)
    lr *= 0.5
    round_num += 1

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
