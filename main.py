import torch.nn as nn
import torch.optim as optim

from src.config import Config
import src.data_loader as data_loader
import src.utils as utils
from src.model import setup_model, test_model, train_model, validate_model


def main():
    print(f"Using model: {Config.MODEL_NAME}, Pretrained: {Config.PRETRAINED}\n")

    device = utils.setup_device()

    dataset = data_loader.prepare_dataset()

    train_dataset, validation_dataset, test_dataset = data_loader.split_dataset(
        dataset=dataset
    )

    train_loader, val_loader, test_loader = data_loader.get_data_loaders(
        train_dataset, validation_dataset, test_dataset
    )

    model = setup_model(len(dataset.classes), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.NUMBER_OF_EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}/{Config.NUMBER_OF_EPOCHS}, "
            f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

    print("Training abgeschlossen!")

    utils.save_model(model, Config.MODEL_SAVE_PATH)

    test_loss, test_accuracy = test_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
