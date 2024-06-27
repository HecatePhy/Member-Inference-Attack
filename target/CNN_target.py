import numpy as np
import pandas as pd

import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

class Cifar10Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 10))

    def trainingStep(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def validationStep(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        accuracy = self.accuracy(out, labels)
        return {"loss": loss, "accuracy": accuracy}

    def validationEpochEnd(self, outputs):
        batchLosses = [row["loss"] for row in outputs]
        epochLosses = torch.stack(batchLosses).mean()
        batchAcc = [row["accuracy"] for row in outputs]
        epochAcc = torch.stack(batchAcc).mean()
        return {"loss": epochLosses.item(), "accuracy": epochAcc.item()}

    def forward(self, x):
        return self.network(x)

@torch.no_grad()
def evaluateModel(model, validationLoader):
    model.eval()
    out = [model.validationStep(batch) for batch in validationLoader]
    return model.validationEpochEnd(out)

def trainModel(epochs, lr, model, trainLoader, validationLoader,
        optimizationFunction=torch.optim.SGD):
    optimizer = optimizationFunction(model.parameters(), lr)
    for epoch in range(epochs):
        print(f"training epoch {epoch}")
        model.train()
        trainingLosses = []
        for batch in trainLoader:
            loss = model.trainingStep(batch)
            trainingLosses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"after training epoch {epoch} we get results {evaluateModel(model, validationLoader)}")
    return

def train_cnn():
    dataset = torchvision.datasets.CIFAR10(
            root="../data",
            train=True,
            download=False,
            transform=transforms.ToTensor())

    testset = torchvision.datasets.CIFAR10(
            root="../data",
            train=False,
            download=False,
            transform=transforms.ToTensor())

    batchSize = 200
    trainset, validateset = random_split(dataset, [45000, 5000])
    train = DataLoader(trainset, batchSize, shuffle=True)
    validate = DataLoader(validateset, batchSize, shuffle=True)
    testLoader = DataLoader(testset, batch_size=batchSize, shuffle=False)
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    model = Cifar10Classifier()
    trainModel(50, 0.001, model,
            trainLoader=train, validationLoader=validate,
            optimizationFunction=torch.optim.Adam)

    print(f"validation dataset accuracy: {evaluateModel(model, validate)}")
    print(f"test dataset accuracy: {evaluateModel(model, testLoader)}")

    # save model
    torch.save(model.state_dict(), '../data/target.pth')

if __name__=="__main__":
    train_cnn()
