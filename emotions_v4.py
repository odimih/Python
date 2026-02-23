# 1. Κατεβάζουμε τα αρχεία
from gdown import download

# download(id='10iQQcGN80wqRMjGeItnklsODP54hHydP', output='model_ferplus.pth', quiet=False)
# download(id='1g56Vxvk506MV4mxf3489WI-KBgmKiRLw', output='angry.png', quiet=False)
# download(id='1ej3OzvPL_Itck2v3Atln671l-RfrvCss', output='happy.png', quiet=False)
# download(id='1-2C4lT5WdAXSleGOG0KHtTeYfyDonhJn', output='neutral.png', quiet=False)

# 2. Φορτώνουμε το μοντέλο
import torch
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class ReshapeAndScale255(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        if len(x.shape) == 2: x = x.unsqueeze(0)
        return (x.unsqueeze(1) / 255).clamp(0,1)

model = torch.nn.Sequential(
    ReshapeAndScale255(),
    models.shufflenet_v2_x1_0(num_classes = 8)
)
model[1].conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model[1].load_state_dict(torch.load('model_ferplus.pth', map_location=device))
model.eval()
model.to(device)

emotion_labels = [
    "neutral",    # 0
    "happy",      # 1
    "surprise",   # 2
    "sad",        # 3
    "angry",      # 4
    "disgust",    # 5
    "fear",       # 6
    "contempt"    # 7
]

# 3. Φορτώνουμε τη φωτογραφία σαν torch.tensor και τις βοηθητικές συναρτήσεις
from PIL import Image
import numpy as np

def loadImage(filename):
    img = Image.open(filename).convert("L").resize(size=(112,112))
    img_data = np.array(img).astype(np.float32)
    return torch.tensor(img_data, device=device)

def tensorToImage(tensor):
    return Image.fromarray(
        tensor.clamp(0,255).detach().reshape(112,112).cpu().numpy().astype(np.uint8)
    )

# 4. Αλλάζουμε τη φωτογραφία και βλέπουμε το καινούριο συναίσθημα

def sparse_attack(img, target_class, pixel_budget, max_delta=10, steps=100, min_confidence=0.6):
    """
    PGD attack that only modifies the top `pixel_budget` pixels by initial gradient magnitude.
    Stops as soon as the ROUNDED image fools the model with sufficient confidence.
    """
    x0 = img.clone().float()
    target = torch.tensor([target_class], device=device)

    # Select which pixels to touch using the initial gradient
    x_init = x0.clone().requires_grad_(True)
    torch.nn.functional.cross_entropy(model(x_init), target).backward()
    grad_init = x_init.grad.detach()

    mask = torch.zeros_like(grad_init)
    mask.flatten()[grad_init.flatten().abs().topk(pixel_budget).indices] = 1.0

    # PGD only on the selected pixels
    x = x0.clone()
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        torch.nn.functional.cross_entropy(model(x), target).backward()

        x = x.detach() - mask * x.grad.sign()
        x = x.clamp(x0 - max_delta, x0 + max_delta).clamp(0, 255)

        # Early stop: check rounded image with minimum confidence threshold
        with torch.no_grad():
            probs = model(x.round()).softmax(-1).squeeze()
            if probs.argmax().item() == target_class and probs[target_class].item() >= min_confidence:
                break

    return x.detach()


def min_budget_attack(img, target_class, max_delta=10, steps=100, n_iters=15):
    """
    Binary search over pixel budgets, tracking actual L1 distance for every K tried.
    Binary search visits large K (fast convergence) and small K (few pixels) values —
    by recording the minimum L1 across all of them we get the best of both worlds.
    """
    lo, hi = 10, img.numel()
    best_x, best_dist = None, float('inf')

    for _ in range(n_iters):
        mid = (lo + hi) // 2
        candidate = sparse_attack(img, target_class, pixel_budget=mid,
                                  max_delta=max_delta, steps=steps, min_confidence=0.0)
        with torch.no_grad():
            pred = model(candidate.clamp(0, 255).round()).argmax().item()
        if pred == target_class:
            dist = (img.round() - candidate.clamp(0, 255).round()).abs().sum().item()
            if dist < best_dist:
                best_x, best_dist = candidate, dist
            hi = mid  # worked: try smaller K
        else:
            lo = mid  # didn't work: try larger K

    if best_x is None:  # fallback
        best_x = sparse_attack(img, target_class, pixel_budget=img.numel(),
                               max_delta=max_delta, steps=steps, min_confidence=0.0)
    return best_x


def min_distance_avoid(img, avoid_class, max_delta=10, steps=100):
    """
    For 'not X' challenges: try every class except avoid_class and return
    the attack with the lowest L1 distance.
    """
    best_x, best_dist = None, float('inf')
    for target in range(8):
        if target == avoid_class:
            continue
        candidate = min_budget_attack(img, target_class=target, max_delta=max_delta, steps=steps)
        with torch.no_grad():
            pred = model(candidate.clamp(0, 255).round()).argmax().item()
        if pred == target:
            dist = (img.round() - candidate.clamp(0, 255).round()).abs().sum().item()
            if dist < best_dist:
                best_x, best_dist = candidate, dist
    return best_x

# Συνάρτηση compare_images
from IPython.display import display

def compare_images(A, B):
    with torch.no_grad():
        predictionA = model(A.clamp(0,255).round()).squeeze().softmax(-1)
        predictionB = model(B.clamp(0,255).round()).squeeze().softmax(-1)

    display(tensorToImage(A))
    print("predictions A:")
    for emotion, probability in zip(emotion_labels, predictionA):
        print(f"{probability*100:8.1f}% - {emotion}")

    emotionA = emotion_labels[predictionA.argmax()]
    print(f"emotion A = {emotionA}\n")

    display(tensorToImage(B))
    print("predictions B:")
    for emotion, probability in zip(emotion_labels, predictionB):
        print(f"{probability*100:8.1f}% - {emotion}")

    emotionB = emotion_labels[predictionB.argmax()]
    print(f"emotion B = {emotionB}\n")

    distance = (A.round() - B.round()).abs().sum().int().item()
    print(f"distance = {distance}")

# EASY: neutral → happy
easy_first = loadImage('neutral.png')
medium_first = loadImage('angry.png')
hard_second = loadImage('happy.png')
easy_second = min_budget_attack(easy_first, target_class=1)
medium_second = min_budget_attack(medium_first, target_class=1)
hard_first = min_distance_avoid(hard_second, avoid_class=1)

compare_images(easy_first, easy_second)
compare_images(medium_first, medium_second)
compare_images(hard_first, hard_second)


# # 5. Αποθήκευση Απαντήσεων για υποβολή στο site
import json

def toList(tensor):
    return tensor.clamp(0,255).round().int().cpu().tolist()

answers = {
    "easy": {
        "first": toList(easy_first),
        "second": toList(easy_second)
    },
    "medium": {
        "second": toList(medium_second),
    },
    "hard": {
        "first": toList(hard_first),
    }
}
with open("answers.json", "w") as f:
    json.dump(answers, f)

# from google.colab import files
# files.download('answers.json')
