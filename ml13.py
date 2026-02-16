import torch
from torch.utils.data import TensorDataset, DataLoader

# Step 1: Create some sample data
# Features (e.g., inputs)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
# Labels (e.g., targets)
y = torch.tensor([0, 1, 0, 1])

# Step 2: Wrap the tensors into a TensorDataset
dataset = TensorDataset(x, y)
print(f"Dataset length: {len(dataset)}")
print(f"First item: {dataset[0]}") Z
# Step 3: Use DataLoader to iterate through the dataset
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 4: Iterate through the DataLoader
for batch_idx, (features, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Features:\n{features}")
    print(f"Labels:\n{labels}")
