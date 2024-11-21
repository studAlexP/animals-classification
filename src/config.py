class Config:
    DATA_DIR = "data/raw-img"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUMBER_OF_EPOCHS = 10
    MODEL_SAVE_PATH = "model.pth"
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
