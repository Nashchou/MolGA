import os
import torch
import random
import csv
from torch.utils.data import Subset, DataLoader

import os
import torch
import random
from collections import defaultdict
def get_fewshot_indices(global_dataset, indices, k=5, num_tasks=100, save_path='fewshot_idx.pt', seed=42):
    if os.path.exists(save_path):
        print(f'Loading few-shot index list from {save_path}')
        return torch.load(save_path)
    

    raw_dataset = global_dataset.data

    is_valid = raw_dataset.y ** 2 > 0

    
    print(f'Generating new few-shot index list and saving to {save_path}')
    random.seed(seed)

    label_to_indices = defaultdict(list)
    for idx in indices:
        label = int(global_dataset.data.y[idx].item())
        if label ** 2 > 0:  # Only consider valid labels
            label_to_indices[label].append(idx.item())

    for label, idxs in label_to_indices.items():
        if len(idxs) < k:
            print(f"Warning: Label {label} has only {len(idxs)} samples, fewer than k={k}")

    tasks = []
    for _ in range(num_tasks):
        task = []
        for label, idxs in label_to_indices.items():
            sampled = random.sample(idxs, min(k, len(idxs)))
            for sid in sampled:
                actual = int(global_dataset.data.y[sid].item())
                if actual != label:
                    print(f"Error: Index {sid} has label {actual}, expected {label}")
            task.extend(sampled)
        tasks.append(task)

    torch.save(tasks, save_path)
    return tasks
