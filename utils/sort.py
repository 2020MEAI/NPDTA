import os
import pickle
import numpy as np
import torch

root_dir = '../data/Metz_s1'
output_root = '../data_sorted/Metz_s1'

for split in ['training', 'testing', 'validation']:
    log_dir = os.path.join(root_dir, split, 'log')
    output_log_dir = os.path.join(output_root, split, 'log')

    os.makedirs(output_log_dir, exist_ok=True)
    for filename in os.listdir(log_dir):
        if '_x_' in filename:
            x_path = os.path.join(log_dir, filename)
            y_path = x_path.replace('_x', '_y')

            with open(x_path, 'rb') as f:
                x_data = pickle.load(f)

            with open(y_path, 'rb') as f:
                y_data = pickle.load(f)

            sorted_idx = torch.argsort(y_data, descending=True)
            x_sorted = x_data[sorted_idx]
            y_sorted = y_data[sorted_idx]

            x_output_path = os.path.join(output_log_dir, os.path.basename(x_path))
            y_output_path = os.path.join(output_log_dir, os.path.basename(y_path))

            with open(x_output_path, 'wb') as f:
                pickle.dump(x_sorted, f)

            with open(y_output_path, 'wb') as f:
                pickle.dump(y_sorted, f)
