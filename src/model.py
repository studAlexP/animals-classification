import torch
import torch.nn as nn
from torchvision import models

from src.config import Config


def setup_model(number_of_classes: int, device: torch.device) -> nn.Module:
    model_name = Config.MODEL_NAME.lower()
    isPretrained = Config.PRETRAINED

    if model_name == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if isPretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, number_of_classes)
    elif model_name == "alexnet":
        model = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT if isPretrained else None
        )
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, number_of_classes
        )
    else:
        raise ValueError(
            f"Model {Config.MODEL_NAME} is not supported. Choose from 'resnet18' or 'alexnet'."
        )
    return model.to(device)


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(train_loader.dataset)


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct.double() / len(val_loader.dataset)
    return val_loss, val_accuracy


def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct.double() / len(test_loader.dataset)

    return test_loss, test_accuracy
