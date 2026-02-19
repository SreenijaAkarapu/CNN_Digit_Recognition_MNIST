import torch
import torch.nn as nn
import cv2
import os
import pandas as pd

# ---------------- CNN architecture ----------------
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

# ---------------- Load model ----------------
model = SimpleCNN()
model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
model.eval()

# ---------------- Load labels ----------------
labels = pd.read_csv("labels.csv")

correct = 0

for i, row in labels.iterrows():

    img_path = os.path.join("images", row["filename"])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img_resized = cv2.resize(img, (28, 28))
    img_norm = img_resized / 255.0
    tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(1).item()

    true_label = row["label"]

    if pred == true_label:
        correct += 1

    # Show image with prediction
    display = cv2.resize(img, (200, 200))
    cv2.putText(display, f"Pred: {pred}  True: {true_label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("CNN Prediction", display)

    key = cv2.waitKey(0)  # press any key for next image
    if key == 27:  # ESC to stop early
        break

cv2.destroyAllWindows()

accuracy = (correct / (i + 1)) * 100
print(f"Visual test accuracy: {accuracy:.2f}%")
