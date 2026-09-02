import os
import numpy as np
import matplotlib.pyplot as plt


def plot_task1_forgetting(task1_history_dict, save_path="results/task1_accuracy.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tasks = [1, 2, 3, 4, 5]
    acc_matrix = np.array(list(task1_history_dict.values()))
    means = np.mean(acc_matrix, axis=0)
    stds = np.std(acc_matrix, axis=0)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.errorbar(tasks, means, yerr=stds, fmt="-o", color="#1f77b4", ecolor="#d62728",
                 elinewidth=2, capsize=5, capthick=2, linewidth=2.5, markersize=8,
                 label="Naive Sequential M=0 (Mean ± Std)")
    plt.title("Task 1 Accuracy Decay Across Sequential Tasks", fontsize=14, fontweight="bold")
    plt.xlabel("After Training Task N", fontsize=12)
    plt.ylabel("Task 1 Accuracy (%)", fontsize=12)
    plt.xticks(tasks)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_replay_sweep(buffer_sizes, mean_accs, std_accs, joint_acc=None, save_path="results/replay_vs_buffer.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.errorbar(buffer_sizes, mean_accs, yerr=std_accs, fmt="-o", color="#1f77b4",
                 ecolor="#d62728", elinewidth=2, capsize=5, capthick=2,
                 linewidth=2.5, markersize=8, label="Experience Replay (Mean ± Std)")

    if joint_acc is not None:
        try:
            joint_val = float(joint_acc)
            plt.axhline(y=joint_val, color="#2ca02c", linestyle="--", linewidth=2,
                        label=f"Joint Upper Bound ({joint_val:.1f}%)")
        except (ValueError, TypeError):
            pass

    plt.title("Final Average Accuracy (A_5) vs. Replay Buffer Size (M)", fontsize=14, fontweight="bold")
    plt.xlabel("Buffer Size (M)", fontsize=12)
    plt.ylabel("Final Average Accuracy A_5 (%)", fontsize=12)
    plt.xticks(buffer_sizes)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_accuracy_curves(curves_dict, save_path="results/accuracy_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tasks = [1, 2, 3, 4, 5]

    t1_matrix      = np.array([v['t1']      for v in curves_dict.values()])
    current_matrix = np.array([v['current'] for v in curves_dict.values()])
    avg_matrix     = np.array([v['avg']     for v in curves_dict.values()])

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    for matrix, color, label in [
        (t1_matrix,      "#d62728", "Task 1 Acc (forgetting)"),
        (current_matrix, "#2ca02c", "New Task Acc (plasticity)"),
        (avg_matrix,     "#1f77b4", "Cumulative Avg Acc"),
    ]:
        means = np.mean(matrix, axis=0)
        stds  = np.std(matrix,  axis=0)
        ax.errorbar(tasks, means, yerr=stds, fmt="-o", color=color, ecolor=color,
                    elinewidth=1.5, capsize=4, capthick=1.5, linewidth=2.5,
                    markersize=7, label=label, alpha=0.85)

    ax.set_title("Accuracy Curves Across Training Stages", fontsize=14, fontweight="bold")
    ax.set_xlabel("After Training Task N", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(tasks)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11, loc="center right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
