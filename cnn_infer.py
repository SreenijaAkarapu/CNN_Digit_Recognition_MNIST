import torch
import torch.nn as nn
import cv2
import numpy as np
import sys
import time

# ----------------------------
# CNN Architecture (same as training)
# ----------------------------
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

# ----------------------------
# Load trained model
# ----------------------------
model = SimpleCNN()
model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
model.eval()

# ----------------------------
# Check input argument
# ----------------------------
if len(sys.argv) != 2:
    print("Usage: python cnn_infer.py path_to_image")
    sys.exit()

image_path = sys.argv[1]

# ----------------------------
# Load and preprocess image
# ----------------------------
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found!")
    sys.exit()

img = cv2.resize(img, (28, 28))
img = img / 255.0

img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# ----------------------------
# Inference
# ----------------------------
start_time = time.perf_counter()

with torch.no_grad():
    output = model(img)
    prediction = output.argmax(1).item()

end_time = time.perf_counter()

elapsed = (end_time - start_time) * 1000
print("Predicted digit:", prediction)
print("Inference time: {:.6f} ms".format(elapsed))

