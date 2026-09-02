import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset


def evaluate_task(model, test_dataset, device='cpu'):
    """
    Evaluates a model on a given test task dataset (2-class subset).
    Used internally to fill R[i,j] — but the model predicts over all 10 classes.

    Returns:
        accuracy (float): Test accuracy in range [0.0, 1.0].
    """
    model.eval()
    loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_cumulative(model, test_tasks, num_tasks_seen, device='cpu'):
    """
    Evaluates model on the cumulative test set of all tasks seen so far.
    This is the correct class-incremental evaluation: never a 2-class test.

    Args:
        test_tasks: list of all 5 task test datasets
        num_tasks_seen: how many tasks have been trained (1-indexed)

    Returns:
        accuracy (float): Accuracy over all classes seen so far [0.0, 1.0].
    """
    cumulative_set = ConcatDataset(test_tasks[:num_tasks_seen])
    return evaluate_task(model, cumulative_set, device)


def compute_metrics(R):
    """
    Calculates key Continual Learning metrics from the Accuracy Matrix R.
    
    R is a (5 x 5) matrix where R[i, j] is the accuracy on task j after training task i.

    Returns:
        A_5 (float): Final Average Accuracy across all 5 tasks (0 - 100%).
        BWT (float): Backward Transfer (measuring forgetting vs improvement).
        F_5 (float): Average Forgetting across tasks 1 to 4.
    """
    T = R.shape[0]  # T = 5 tasks

    # 1. Final Average Accuracy (A_5): Average of the bottom row of matrix R
    A_5 = np.mean(R[T - 1, :]) * 100.0

    # 2. Backward Transfer (BWT): Measures accuracy drop on task i from when it was first learned
    bwt_sum = 0.0
    for i in range(T - 1):
        bwt_sum += (R[T - 1, i] - R[i, i])
    BWT = (bwt_sum / (T - 1)) * 100.0

    # 3. Average Forgetting (F_5): Maximum historical accuracy minus final accuracy
    forgetting_list = []
    for j in range(T - 1):
        max_acc_j = np.max(R[:T - 1, j])
        final_acc_j = R[T - 1, j]
        forgetting_list.append(max_acc_j - final_acc_j)
    F_5 = np.mean(forgetting_list) * 100.0

    return A_5, BWT, F_5