import numpy as np
from torch.utils.data import SubsetRandomSampler
import os
from torch.utils.data import Dataset, DataLoader
import torch
import random
from torchvision import transforms
from collections import OrderedDict
import torch.nn as nn


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if "pool" in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif "deconv" in layer_name:
            transposeConv2d = nn.ConvTranspose2d(
                in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
            )
            layers.append((layer_name, transposeConv2d))
            if "relu" in layer_name:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))
            elif "leaky" in layer_name:
                layers.append(
                    ("leaky_" + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True))
                )
        elif "conv" in layer_name:
            conv2d = nn.Conv2d(
                in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
            )
            layers.append((layer_name, conv2d))
            if "relu" in layer_name:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))
            elif "leaky" in layer_name:
                layers.append(
                    ("leaky_" + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True))
                )
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))


def set_seed(seed: int):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(mode=True)


class TempConvLSTMDataset(Dataset):
    def __init__(self, folder_path, input_seq_len=10, output_seq_len=10, transform=None):

        self.folder_path = folder_path
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.total_len = input_seq_len + output_seq_len
        self.transform = transform

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
        data = np.stack(data_list)
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_start = idx
        input_end = input_start + self.input_seq_len
        output_start = input_end
        output_end = output_start + self.output_seq_len
        input_seq = self.data[input_start:input_end]
        output_seq = self.data[output_start:output_end]

        input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(1)
        output_seq = torch.tensor(output_seq, dtype=torch.float32).unsqueeze(1)

        if self.transform:
            input_seq = self.transform(input_seq)
            output_seq = self.transform(output_seq)

        return input_seq, output_seq


class TempDataset:
    def __init__(
        self,
        folder_path="data/tmean_interp_final/npy",
        batch_size=32,
        input_seq_len=10,
        output_seq_len=10,
        split_test=0.1,
        split_val=0.1,
        seed: int = 42,
        workers=0,
        rezise=(64, 64),
    ):

        self.folder_path = folder_path
        self.batch_size = batch_size
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.split_test = split_test
        self.split_val = split_val
        self.seed = seed
        self.workers = workers
        self.rezise = rezise

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.rezise),
            ]
        )

    def get_loaders(self):
        train_dataset = TempConvLSTMDataset(
            self.folder_path, self.input_seq_len, self.output_seq_len, transform=self.transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

        train_sample, val_sample, test_sample = get_test_val_set(
            train_dataset, self.split_test, self.split_val, self.seed
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sample,
            num_workers=self.workers,
        )

        val_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=val_sample,
            num_workers=self.workers,
        )

        test_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=test_sample,
            num_workers=self.workers,
        )

        return train_loader, val_loader, test_loader


def get_test_val_set(dst_train, split_test=0.1, split_val=0.1, seed: int = 42):

    set_seed(seed)

    indices = list(range(len(dst_train)))
    np.random.shuffle(indices)
    split_test = int(np.floor(split_test * len(dst_train)))
    split_val = int(np.floor(split_val * len(dst_train)))
    train_indices, val_indices, test_indices = (
        indices[split_test + split_val :],
        indices[:split_val],
        indices[split_val : split_test + split_val],
    )

    train_sample = SubsetRandomSampler(train_indices)
    val_sample = SubsetRandomSampler(val_indices)
    test_sample = SubsetRandomSampler(test_indices)

    return train_sample, val_sample, test_sample


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
