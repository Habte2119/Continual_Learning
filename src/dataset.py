import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_split_mnist(data_root="./data"):
    """
    Downloads MNIST automatically and splits it into 5 sequential tasks of 2 classes each.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Downloads MNIST automatically to ./data
    full_train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    full_test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    # 5 Sequential tasks: (0,1), (2,3), (4,5), (6,7), (8,9)
    task_classes = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]
    ]

    train_tasks = []
    test_tasks = []

    for classes in task_classes:
        train_indices = [
            i for i, label in enumerate(full_train_set.targets) 
            if label.item() in classes
        ]
        train_tasks.append(Subset(full_train_set, train_indices))

        test_indices = [
            i for i, label in enumerate(full_test_set.targets) 
            if label.item() in classes
        ]
        test_tasks.append(Subset(full_test_set, test_indices))

    return train_tasks, test_tasks, full_train_set, full_test_set