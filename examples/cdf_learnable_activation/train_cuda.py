import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import KFold
import numpy as np

from cdflearnableactivation import ModifiedCNN

def load_dataset():
    # Define the transformation pipeline for the dataset
    transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.ToTensor()
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return trainset

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

def evaluate_model(model, test_loader, device):
    model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    return accuracy, precision

def k_fold_validation(dataset, k=5, num_epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kf = KFold(n_splits=k, shuffle=True, random_state=None)
    results = []
    for train_index, test_index in kf.split(dataset):
        train_sampler = SubsetRandomSampler(train_index)
        test_sampler = SubsetRandomSampler(test_index)
        train_loader = DataLoader(dataset, batch_size=10, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=10, sampler=test_sampler)
        model = ModifiedCNN(num_classes=10, device=device)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device)
        accuracy, precision = evaluate_model(model, test_loader, device)
        results.append((accuracy, precision))
        print(f"Fold Results - Accuracy: {accuracy}, Precision: {precision}")
    average_accuracy = np.mean([x[0] for x in results])
    average_precision = np.mean([x[1] for x in results])
    print(f"K-Fold Validation Results - Average Accuracy: {average_accuracy}, Average Precision: {average_precision}")

if __name__ == "__main__":
    dataset = load_dataset()
    k_fold_validation(dataset)
