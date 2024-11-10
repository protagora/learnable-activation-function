import sys
import os
sys.path.append(os.path.abspath('../../modules'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import KFold
import numpy as np

from cdflearnableactivation import ModifiedCNN

def evaluate_model(model, test_loader):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())
            true_labels.extend(labels.numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')

    return accuracy, precision

transform = transforms.Compose([
    transforms.Resize((225, 225)),
    transforms.ToTensor()
])

def load_dataset():
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return trainset

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

def k_fold_validation(dataset, k=5, num_epochs=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=None)
    results = []

    for train_index, test_index in kf.split(dataset):
        train_sampler = SubsetRandomSampler(train_index)
        test_sampler = SubsetRandomSampler(test_index)

        train_loader = DataLoader(dataset, batch_size=10, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=10, sampler=test_sampler)

        model = ModifiedCNN(num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
        accuracy, precision = evaluate_model(model, test_loader)
        results.append((accuracy, precision))
        print(f"Fold Results - Accuracy: {accuracy}, Precision: {precision}")

    average_accuracy = np.mean([x[0] for x in results])
    average_precision = np.mean([x[1] for x in results])
    print(f"K-Fold Validation Results - Average Accuracy: {average_accuracy}, Average Precision: {average_precision}")

if __name__ == "__main__":
    dataset = load_dataset()
    k_fold_validation(dataset)
