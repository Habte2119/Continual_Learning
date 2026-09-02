import torch
import torch.nn as nn


class SplitMNISTMLP(nn.Module):
    """
    A 3-layer Multi-Layer Perceptron (MLP) for Split-MNIST.

    Architecture:
    - Input Layer: 784 nodes (28x28 flattened MNIST image)
    - Hidden Layer 1: 256 nodes + ReLU activation
    - Hidden Layer 2: 128 nodes + ReLU activation
    - Output Head: 10 logits (one per digit class 0–9)
    """

    def __init__(self, input_dim=784, hidden_dim1=256, hidden_dim2=128, num_classes=10):
        super(SplitMNISTMLP, self).__init__()

        # Feature extractor backbone (shared across all tasks)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
        )

        # Single-head output classifier for digits 0-9
        self.classifier = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        # Flatten image from (B, 1, 28, 28) to (B, 784)
        x = x.view(x.size(0), -1)

        # Extract features through shared hidden layers
        features = self.feature_extractor(x)

        # Map hidden representations to 10 class logits
        logits = self.classifier(features)
        return logits