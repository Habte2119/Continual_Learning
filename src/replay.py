import random
import torch


class ReplayBuffer:
    """
    Experience Replay Buffer for Continual Learning.
    Stores up to capacity M total exemplars balanced across observed tasks.
    """
    def __init__(self, capacity=200):
        self.capacity = capacity
        self.buffer_x = []
        self.buffer_y = []

    def __len__(self):
        return len(self.buffer_x)

    def add_samples(self, dataset, samples_per_task):
        """
        Extracts random samples from a completed task dataset and appends to buffer.
        """
        if self.capacity == 0 or samples_per_task == 0:
            return

        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        all_x, all_y = next(iter(loader))

        num_to_add = min(samples_per_task, len(all_x))
        indices = random.sample(range(len(all_x)), num_to_add)

        for idx in indices:
            if len(self.buffer_x) >= self.capacity:
                # If buffer is full, randomly replace an existing sample
                replace_idx = random.randint(0, self.capacity - 1)
                self.buffer_x[replace_idx] = all_x[idx]
                self.buffer_y[replace_idx] = all_y[idx]
            else:
                self.buffer_x.append(all_x[idx])
                self.buffer_y.append(all_y[idx])

    def sample(self, batch_size):
        """
        Randomly samples a mini-batch of past exemplars from the buffer.
        """
        if len(self.buffer_x) == 0:
            return None, None

        actual_batch_size = min(batch_size, len(self.buffer_x))
        indices = random.sample(range(len(self.buffer_x)), actual_batch_size)

        batch_x = torch.stack([self.buffer_x[i] for i in indices])
        batch_y = torch.tensor([self.buffer_y[i] for i in indices])

        return batch_x, batch_y