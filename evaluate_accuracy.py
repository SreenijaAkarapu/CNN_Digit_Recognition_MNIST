
import torch
import torch.nn as nn
import cv2
import os
import pandas as pd

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN()
model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
model.eval()

labels = pd.read_csv("labels.csv")

correct = 0
total = len(labels)

for _, row in labels.iterrows():
    img_path = os.path.join("images", row["filename"])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()

    if pred == row["label"]:
        correct += 1

accuracy = (correct / total) * 100
print(f"Accuracy on 100 MNIST test images: {accuracy:.2f}%")
