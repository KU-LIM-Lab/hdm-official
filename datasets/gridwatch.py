import pandas as pd
import numpy as np
import torch

from datasets.utils import InstanceStandardize

class GridwatchDataset(torch.utils.data.Dataset):
    def __init__(self, phase='train', train_test_ratio=0.8, seed=87):
        super().__init__()

        self.phase = phase
        self.scaler = InstanceStandardize()
        self.seed = seed
        self.num_points = 288
        self.num_samples = 1013
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test \
            = self._get_gridwatch_dataset(train_test_ratio)

    def __len__(self):
        if self.phase == 'train':
            return self.x_train.shape[0]
        elif self.phase == 'valid':
            return self.x_valid.shape[0]
        else:
            return self.x_test.shape[0]

    def __getitem__(self, idx):

        if self.phase == 'train':
            x, y = self.x_train, self.y_train
        elif self.phase == 'valid':
            x, y = self.x_valid, self.y_valid
        else:
            x, y = self.x_test, self.y_test

        return x[idx], y[idx]

    def _get_gridwatch_dataset(self, train_test_ratio):
        np.random.seed(self.seed)

        data = pd.read_csv('./data/gridwatch_clean.csv', index_col=0)

        x_full = torch.arange(self.num_points, dtype=torch.float).unsqueeze(-1)
        demand_values = data[' demand']

        y_full = np.zeros((self.num_samples, self.num_points, 1), dtype=np.float32)
        for i in range(self.num_samples):
            y_full[i] = demand_values[i*self.num_points:(i+1)*self.num_points].values.reshape(-1,1)

        y_full = torch.tensor(y_full)

        num_train = int(self.num_samples * train_test_ratio)
        train_indices = np.random.choice(self.num_samples, num_train, replace=False)
        eval_indices = np.setdiff1d(np.arange(y_full.shape[0]), train_indices)

        x_train = x_full.repeat(num_train, 1, 1)
        y_train = y_full[train_indices]

        x_test = x_full.repeat(self.num_samples - num_train, 1, 1)
        y_test = y_full[eval_indices]

        y_train = self.scaler.fit_transform(y_train)
        y_test = self.scaler.fit_transform(y_test)

        num_test = int(y_test.shape[0] * 0.5)

        x_valid = x_test[:num_test]
        y_valid = y_test[:num_test]

        x_test = x_test[num_test:]
        y_test = y_test[num_test:]

        return x_train, y_train, x_valid, y_valid, x_test, y_test
