import os.path

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# # Plotting
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# Dataset Initialization
from torch.utils.data import random_split
import ChordDataset
from torch.utils.data import DataLoader


class ChordCNN(nn.Module):

    def __init__(self):
        super(ChordCNN, self).__init__()

        # Convolutional block #1
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)

        # Convolutional block #2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)

        # Convolutional block #3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2,2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 7)


        self.dataset_loaded = False



    def forward(self, x):
        # Each layer has the convolutional layer, a ReLU activation function, and then a max pooling layer.
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (16, 64, 64)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (32, 32, 32)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (64, 16, 16)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



def training_loop(epochs, model, loader, criterion, optimizer, device="cpu"):

    # fig, ax = plt.subplots()

    train_losses = []
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("Loss")

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0

        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward Pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


            train_losses.append(loss.item())

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # plt.legend()
    # plt.grid(True)
    # plt.plot(train_losses)
    # plt.show()

def get_accuracy(model, loader, device="cpu") -> None:
    correct = 0
    total = 0

    with torch.no_grad():
        model.eval()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)


    print(f"Accuracy: {100 * correct / total:.2f}%")

def load_model(model, device="cpu"):

    path = input("Please Enter Filename: ")
    if path == "":
        print("Model failed to load. Incomplete path.")
        return
    if not os.path.exists(path):
        print("Model failed to load. File does not exist.")
        return

    model.load_state_dict(torch.load(path, device))
    model.to(device)
    model.eval()

    print("Model has been successfully loaded.")

def initialize_dataset():
    dataset = ChordDataset.ChordDataset("data/")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    return train_loader, test_loader