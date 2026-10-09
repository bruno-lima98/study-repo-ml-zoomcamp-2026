# %%
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu")

print(torch.__version__)

# %%
evaluation_transform = transforms.Compose([
    transforms.Resize((200, 200), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_transform = evaluation_transform

# %%
DATA_DIR = "./data"

train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
evaluation_dataset = datasets.ImageFolder(f"{DATA_DIR}/test", transform=evaluation_transform)

print(train_dataset.class_to_idx) 
print(len(train_dataset), len(evaluation_dataset)) 

# %%
loader_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    shuffle=True,
    num_workers=0,
    generator=loader_generator,
)
evaluation_loader = DataLoader(
    evaluation_dataset,
    batch_size=20,
    shuffle=False,
    num_workers=0,
)

# %%
class HairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.hidden = nn.Linear(32 * 99 * 99, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv(x)))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.hidden(x))
        return self.output(x)


model = HairModel().to(device)

# %%
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.002,
    momentum=0.8,
)

# %%
sum(p.numel() for p in model.parameters())

# %%
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # [20] -> [20, 1], float

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += batch_size

    return running_loss / total, correct / total


def train_model(model, train_loader, evaluation_loader, criterion, optimizer, num_epochs):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "evaluation_loss": [],
        "evaluation_accuracy": [],
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        eval_loss, eval_acc = evaluate(model, evaluation_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["evaluation_loss"].append(eval_loss)
        history["evaluation_accuracy"].append(eval_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"eval loss {eval_loss:.4f} acc {eval_acc:.4f}"
        )

    return history


# %%
history_baseline = train_model(
    model, train_loader, evaluation_loader, criterion, optimizer, num_epochs=10
)

with open("history_baseline.json", "w") as f:
    json.dump(history_baseline, f, indent=4)

# %%
with open("history_baseline.json") as f:
    history = json.load(f)

print(history["train_accuracy"])
print(round(float(np.median(history["train_accuracy"])), 2))

# %%
print(history["train_loss"])
print(round(float(np.std(history["train_loss"], ddof=0)), 3))

# %%
train_transform_augmented = transforms.Compose([
    transforms.Resize((200, 200), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomRotation(
        50,
        interpolation=transforms.InterpolationMode.NEAREST,
    ),
    transforms.RandomResizedCrop(
        200,
        scale=(0.9, 1.0),
        ratio=(0.9, 1.1),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset.transform = train_transform_augmented

# %%
history_augmented = train_model(
    model, 
    train_loader, 
    evaluation_loader, 
    criterion, 
    optimizer, 
    num_epochs=10
)

with open("history_augmented.json", "w") as f:
    json.dump(history_augmented, f, indent=4)

# %%
with open("history_augmented.json") as f:
    history_aug = json.load(f)

print(history_aug["evaluation_loss"])
print(round(float(np.mean(history_aug["evaluation_loss"])), 3))

# %%
last_five = history_aug["evaluation_accuracy"][-5:]

print(last_five)
print(round(float(np.mean(last_five)), 2))
# %%
