import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def diagnose_h1_vs_h2(model, task1_train_dataset, task1_test_dataset, device='cpu', epochs=5, lr=0.01):
    """
    Performs a Linear Probe on the baseline model after all 5 tasks are trained.

    1. Evaluates original model on Task 1 (Measures combined impact of H1 and H2).
    2. Freezes feature_extractor, replaces output classifier with a fresh linear head.
    3. Retrains only the new head on Task 1 data.
    4. Evaluates probed model on Task 1 (Isolates H1: if accuracy recovers, features were intact).
    """
    # 1. Baseline performance on Task 1 (as-is)
    model.eval()
    t1_loader_test = DataLoader(task1_test_dataset, batch_size=128, shuffle=False)
    
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in t1_loader_test:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc_before_probe = (correct / total) * 100.0

    # 2. Freeze feature extractor layers
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    # Store original classification head weights
    original_classifier = model.classifier

    # Replace classifier with a fresh linear probe head
    model.classifier = nn.Linear(128, 10).to(device)

    # 3. Train linear probe on Task 1 training data
    t1_loader_train = DataLoader(task1_train_dataset, batch_size=128, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for x, y in t1_loader_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    # 4. Evaluate probed model performance on Task 1
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in t1_loader_test:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc_after_probe = (correct / total) * 100.0

    # Restore original classifier state and unfreeze features for integrity
    model.classifier = original_classifier
    for param in model.feature_extractor.parameters():
        param.requires_grad = True

    return {
        'original_task1_acc': acc_before_probe,
        'probed_task1_acc': acc_after_probe,
        'feature_recovery_delta': acc_after_probe - acc_before_probe
    }