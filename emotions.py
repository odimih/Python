# 1. Κατεβάζουμε τα αρχεία
from gdown import download

# download(id='10iQQcGN80wqRMjGeItnklsODP54hHydP', output='model_ferplus.pth', quiet=False)
# download(id='1g56Vxvk506MV4mxf3489WI-KBgmKiRLw', output='angry.png', quiet=False)
# download(id='1ej3OzvPL_Itck2v3Atln671l-RfrvCss', output='happy.png', quiet=False)
# download(id='1-2C4lT5WdAXSleGOG0KHtTeYfyDonhJn', output='neutral.png', quiet=False)

# 2. Φορτώνουμε το μοντέλο
import torch
from torchvision import models

if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")
print(f"Using device: {device}")

class ReshapeAndScale255(torch.nn.Module):
    def __init__(self): 
        super().__init__( )   
    def forward(self, x):
        if len(x.shape) == 2: 
            x = x.unsqueeze(0)
        return (x.unsqueeze(1) / 255).clamp(0,1)

model = torch.nn.Sequential(
    ReshapeAndScale255(),
    models.shufflenet_v2_x1_0(num_classes = 8)
)
model[1].conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model[1].load_state_dict(torch.load('model_ferplus.pth', map_location=device))
model.eval()

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
    return torch.tensor(img_data)

def tensorToImage(tensor):
    return Image.fromarray(
        tensor.clamp(0,255).detach().reshape(112,112).numpy().astype(np.uint8)
    )

# 4. Αλλάζουμε τη φωτογραφία και βλέπουμε το καινούριο συναίσθημα

# Targeted sparse PGD: push image toward 'happy' (class 1)
# Only the top `top_fraction` pixels by gradient magnitude are updated each step,
# reducing L1 distance while preserving reliable convergence.
def make_happy_version(img, steps=500, step_size=0.1, epsilon=3.0, top_fraction=0.30):
    x0 = img.clone().float()
    x = x0.clone()

    for _ in range(steps):
        x.requires_grad_(True)

        x_scaled = (x / 255).unsqueeze(0).unsqueeze(0)
        out = model[1](x_scaled)

        # stop as soon as happy becomes the top class
        if out.argmax() == 1:
            break

        target = torch.tensor([1])
        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()
        grad = x.grad

        # sparse sign step: only update top-fraction pixels by gradient magnitude
        threshold = grad.abs().quantile(1 - top_fraction)
        mask = (grad.abs() >= threshold).float()
        x = x - step_size * grad.sign() * mask
        x = torch.clamp(x, x0 - epsilon, x0 + epsilon)
        x = x.detach().clamp(0, 255)

    return x

def make_not_happy(img, steps=500, step_size=0.1, epsilon=2.0, top_fraction=0.30):
    x0 = img.clone().float()
    x = x0.clone()

    # pick the highest-confidence non-happy class as the target
    with torch.no_grad():
        x_scaled = (x / 255).unsqueeze(0).unsqueeze(0)
        probs = model[1](x_scaled).squeeze().softmax(-1).clone()
        probs[1] = -1
        target_class = int(probs.argmax().item())

    target = torch.tensor([target_class])

    for _ in range(steps):
        x.requires_grad_(True)

        x_scaled = (x / 255).unsqueeze(0).unsqueeze(0)
        out = model[1](x_scaled)

        # stop as soon as any non-happy class becomes the top class
        if out.argmax() != 1:
            break

        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()
        grad = x.grad

        # sparse sign step: only update top-fraction pixels by gradient magnitude
        threshold = grad.abs().quantile(1 - top_fraction)
        mask = (grad.abs() >= threshold).float()
        x = x - step_size * grad.sign() * mask
        x = torch.clamp(x, x0 - epsilon, x0 + epsilon)
        x = x.detach().clamp(0, 255)

    return x

# Συνάρτηση compare_images
from IPython.display import display

def compare_images(A, B):
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
easy_second = make_happy_version(easy_first)
medium_second = make_happy_version(medium_first, epsilon=4.0)
hard_first = make_not_happy(hard_second)

compare_images(easy_first, easy_second)
compare_images(medium_first, medium_second)
compare_images(hard_first, hard_second)


# # 5. Αποθήκευση Απαντήσεων για υποβολή στο site
import json

def toList(tensor):
    return tensor.clamp(0,255).round().int().tolist()

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