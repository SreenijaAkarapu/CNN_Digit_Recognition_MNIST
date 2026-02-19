import torch
import torch.nn as nn

# Same CNN architecture used during training
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

# Load trained model
model = SimpleCNN()
model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
model.eval()

# Dummy input (required for ONNX export)
dummy_input = torch.randn(1, 1, 28, 28)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "cnn_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("ONNX model exported successfully!")
import torch
import torch.nn as nn

# Same CNN architecture used during training
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

# Load trained model
model = SimpleCNN()
model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
model.eval()

# Dummy input (required for ONNX export)
dummy_input = torch.randn(1, 1, 28, 28)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "cnn_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("ONNX model exported successfully!")
