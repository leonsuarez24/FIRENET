import numpy as np
from torch.utils.data import SubsetRandomSampler
import os
from torch.utils.data import Dataset, DataLoader
import torch


class TempConvLSTMDataset(Dataset):
    def __init__(self, folder_path, input_seq_len=10, output_seq_len=10):

        self.folder_path = folder_path
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.total_len = input_seq_len + output_seq_len

        self.data = self.load_data()

        self.num_samples = len(self.data) - self.total_len + 1

    def load_data(self):
        file_paths = sorted(
            [
                os.path.join(self.folder_path, f)
                for f in os.listdir(self.folder_path)
                if f.endswith(".npy")
            ]
        )
        data_list = [np.load(file) for file in file_paths]
        data = torch.tensor(np.stack(data_list), dtype=torch.float32)
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_start = idx
        input_end = input_start + self.input_seq_len
        output_start = input_end
        output_end = output_start + self.output_seq_len
        input_seq = self.data[input_start:input_end].unsqueeze(1)
        output_seq = self.data[output_start:output_end].unsqueeze(1)

        return input_seq, output_seq


def get_validation_set(dst_train, split: float = 0.1):

    indices = list(range(len(dst_train)))
    np.random.shuffle(indices)
    split = int(np.floor(split * len(dst_train)))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sample = SubsetRandomSampler(train_indices)
    val_sample = SubsetRandomSampler(val_indices)

    return train_sample, val_sample


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_metrics(save_path):

    images_path = save_path + "/images"
    model_path = save_path + "/model"
    metrics_path = save_path + "/metrics"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    return images_path, model_path, metrics_path


def save_npy_metric(file, metric_name):

    with open(f"{metric_name}.npy", "wb") as f:
        np.save(f, file)
