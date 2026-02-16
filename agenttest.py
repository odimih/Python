import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) device")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")

# Create a test tensor and move it to the appropriate device
x = torch.rand(5, 3).to(device)
print("Tensor device:", x.device)
print("Tensor content:", x)