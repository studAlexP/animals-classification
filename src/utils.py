import torch


def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
