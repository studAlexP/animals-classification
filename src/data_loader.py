from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import Config


def prepare_dataset():
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(Config.DATA_DIR, transform=transform)

    return dataset


def split_dataset(dataset):
    train_size = int(0.7 * len(dataset))
    validation_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [train_size, validation_size, test_size]
    )

    return train_dataset, validation_dataset, test_dataset


def get_data_loaders(train_dataset, validation_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        validation_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
