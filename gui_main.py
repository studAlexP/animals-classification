import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.simpledialog import askstring

import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
from sklearn.metrics import f1_score
from torchvision import transforms

import src.utils as utils
from src.config import Config
from src.model import setup_model

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def load_model():
    device = utils.setup_device()
    model = setup_model(number_of_classes=len(Config.CLASSES), device=device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    return model, device


def classify_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        class_name = Config.CLASSES[predicted_class.item()]
        return class_name, predicted_class.item(), confidence.item() * 100


true_labels = []
predicted_labels = []


def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    if file_path:
        try:
            true_class_name = Config.CLASSES[0]

            true_class_name = askstring(
                "True Label", "Enter the true label of the image :"
            ).lower()
            if true_class_name not in Config.CLASSES:
                messagebox.showerror(
                    "Error", "Invalid label. Please enter a valid label."
                )
                return

            true_class_index = Config.CLASSES.index(true_class_name)
            true_labels.append(true_class_index)

            class_name, predicted_class, confidence = classify_image(file_path)
            predicted_labels.append(predicted_class)

            image = Image.open(file_path)
            image.thumbnail((300, 300))
            img = ImageTk.PhotoImage(image)
            image_label.config(image=img)
            image_label.image = img
            result_label.config(
                text=f"Predicted: {class_name} ({confidence:.2f}% confidence)"
            )

            f1 = f1_score(true_labels, predicted_labels, average="weighted")
            f1_label.config(text=f"F1 Score: {f1:.2f}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to classify the image: {str(e)}")


root = tk.Tk()
root.title("Image Classifier")
root.geometry("400x500")

select_button = tk.Button(root, text="Select Image", command=open_image)
select_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=10)

f1_label = tk.Label(root, text="F1 Score: N/A", font=("Helvetica", 14))
f1_label.pack(pady=10)

model, device = load_model()
root.mainloop()
