import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import SplitMNISTMLP
from src.dataset import get_split_mnist
from src.evaluate import evaluate_task, evaluate_cumulative, compute_metrics
from src.diagnostics import diagnose_h1_vs_h2
from src.replay import ReplayBuffer
from utils.plotting import plot_task1_forgetting, plot_replay_sweep

SEEDS = [42, 123, 999]
BUFFER_SIZES = [0, 50, 100, 200]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_task_with_replay(model, task_dataset, buffer, epochs=5, lr=0.001, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(task_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for x_curr, y_curr in loader:
            x_curr, y_curr = x_curr.to(DEVICE), y_curr.to(DEVICE)

            if len(buffer) > 0:
                x_buf, y_buf = buffer.sample(batch_size=batch_size // 2)
                if x_buf is not None:
                    x_buf, y_buf = x_buf.to(DEVICE), y_buf.to(DEVICE)
                    x_curr = torch.cat([x_curr, x_buf], dim=0)
                    y_curr = torch.cat([y_curr, y_buf], dim=0)

            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x_curr), y_curr)
            loss.backward()
            optimizer.step()


def run_joint_baseline(train_set, test_tasks, epochs=10, lr=0.001):
    model = SplitMNISTMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=128, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            optimizer.step()

    return evaluate_cumulative(model, test_tasks, num_tasks_seen=5, device=DEVICE) * 100.0


def main():
    os.makedirs("results", exist_ok=True)
    print(f"=== Starting Experiments on Device: {DEVICE} ===")

    all_sweep_results = {m: [] for m in BUFFER_SIZES}
    t1_forgetting_curves = {}
    accuracy_curves = {M: {} for M in BUFFER_SIZES}
    diagnostic_results = []
    joint_accs = []

    for seed in SEEDS:
        print(f"\n--- Running Seed: {seed} ---")
        set_seed(seed)
        train_tasks, test_tasks, full_train_set, full_test_set = get_split_mnist()

        # 1. Joint Training (Upper Bound)
        joint_acc = run_joint_baseline(full_train_set, test_tasks)
        joint_accs.append(joint_acc)
        print(f"[Seed {seed}] Joint Upper Bound Accuracy (A_5): {joint_acc:.2f}%")

        # 2. Replay Buffer Sweep
        for M in BUFFER_SIZES:
            set_seed(seed)
            model = SplitMNISTMLP().to(DEVICE)
            buffer = ReplayBuffer(capacity=M)

            R = np.zeros((5, 5))
            t1_acc_history = []
            current_task_acc_history = []
            cumulative_avg_history = []
            samples_per_task = M // 5 if M > 0 else 0

            for i in range(5):
                train_task_with_replay(model, train_tasks[i], buffer)

                if M > 0:
                    buffer.add_samples(train_tasks[i], samples_per_task)

                for j in range(5):
                    R[i, j] = evaluate_task(model, test_tasks[j], DEVICE)

                t1_acc_history.append(
                    evaluate_cumulative(model, test_tasks, num_tasks_seen=i + 1, device=DEVICE) * 100.0
                )
                current_task_acc_history.append(R[i, i] * 100.0)
                cumulative_avg_history.append(np.mean(R[i, :i + 1]) * 100.0)

            A_5, BWT, F_5 = compute_metrics(R)
            all_sweep_results[M].append((A_5, BWT, F_5))

            # Save R matrix
            np.savetxt(f"results/R_matrix_M{M}_seed{seed}.csv", R, delimiter=",",
                       fmt="%.4f", header="task0,task1,task2,task3,task4", comments="")


            if M == 0:
                t1_forgetting_curves[f"Seed {seed} (M=0)"] = t1_acc_history

                # Sanity check: Task 1 acc after Task 5 must be < 50% (10-class eval)
                task1_acc_after_task5 = R[4, 0] * 100.0
                print(f"[Sanity Check M=0, Seed {seed}] Task 1 Acc after Task 5: {task1_acc_after_task5:.2f}%")
                assert task1_acc_after_task5 < 50.0, (
                    f"CRITICAL ERROR: Task 1 accuracy after Task 5 is {task1_acc_after_task5:.2f}% (> 50%). "
                    f"Evaluation is using 2-class setup instead of 10-class!"
                )
                print("  -> Passed: Confirmed single-head class-incremental evaluation.")

                # 3. Diagnostic Probe (H1 vs H2)
                diag = diagnose_h1_vs_h2(model, train_tasks[0], test_tasks[0], device=DEVICE)
                diag['seed'] = seed
                diagnostic_results.append(diag)
                print(f"[Seed {seed}] Diagnostic (M=0): Original Task 1 Acc = {diag['original_task1_acc']:.2f}%, Probed Task 1 Acc = {diag['probed_task1_acc']:.2f}%")

            accuracy_curves[M][f"Seed {seed}"] = {
                't1':      t1_acc_history,
                'current': current_task_acc_history,
                'avg':     cumulative_avg_history,
            }

            print(f"[Seed {seed}] Buffer M={M:3d} -> A_5: {A_5:.2f}%, BWT: {BWT:.2f}%, F_5: {F_5:.2f}%")

    # Aggregate results
    print("\n=== Final Aggregated Results Across Seeds ===")
    means_acc, stds_acc = [], []
    with open("results/replay_sweep_metrics.csv", "w") as f:
        f.write("Buffer_Size_M,Mean_A5,Std_A5,Mean_BWT,Std_BWT\n")
        for M in BUFFER_SIZES:
            a5_vals  = [res[0] for res in all_sweep_results[M]]
            bwt_vals = [res[1] for res in all_sweep_results[M]]
            m_a5, s_a5   = np.mean(a5_vals),  np.std(a5_vals)
            m_bwt, s_bwt = np.mean(bwt_vals), np.std(bwt_vals)
            means_acc.append(m_a5)
            stds_acc.append(s_a5)
            f.write(f"{M},{m_a5:.2f},{s_a5:.2f},{m_bwt:.2f},{s_bwt:.2f}\n")
            print(f"M={M:3d} | A_5: {m_a5:.2f}% +/- {s_a5:.2f}% | BWT: {m_bwt:.2f}% +/- {s_bwt:.2f}%")

    # Save diagnostic CSV
    with open("results/diagnostic_h1_vs_h2.csv", "w") as f:
        f.write("seed,original_task1_acc,probed_task1_acc,feature_recovery_delta\n")
        for d in diagnostic_results:
            f.write(f"{d['seed']},{d['original_task1_acc']:.2f},{d['probed_task1_acc']:.2f},{d['feature_recovery_delta']:.2f}\n")

    # Save plots
    mean_joint = float(np.mean(joint_accs))
    plot_task1_forgetting(t1_forgetting_curves, "results/task1_accuracy.png")
    plot_replay_sweep(BUFFER_SIZES, means_acc, stds_acc,
                      joint_acc=mean_joint, save_path="results/replay_vs_buffer.png")
    print("\nSuccess! Results, CSVs, and plots generated in results/")


if __name__ == "__main__":
    main()
