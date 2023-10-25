import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sktime.datasets import load_from_arff_to_dataframe

from datasets.utils import DatasetStandardizer

class MelbourneDataset(torch.utils.data.Dataset):
    def __init__(self, phase='train', test_ratio=0.2, seed=42):
        super(MelbourneDataset, self).__init__()

        self.phase = phase
        self.scaler = DatasetStandardizer()
        self.seed = seed
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test \
            = self._get_melbourne_dataset(test_ratio)
        

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

    def _get_melbourne_dataset(self, test_ratio):
        np.random.seed(self.seed)
        test_file_path = './data/MelbournePedestrian_TEST.arff'

        data = load_from_arff_to_dataframe(test_file_path)
        samples = []
        labels = []

        for i in range(data[0].shape[0]):
            samples.append(torch.tensor(data[0]['dim_0'][i]))
            labels.append(data[1][i])

        label = np.array(labels)
        y_full = torch.stack(samples, axis=0)
        ind = torch.isnan(y_full).sum(axis=1) == 0
        y_full = y_full[ind].float()
        label = label[ind.numpy()]

        y = y_full.numpy()
        x_full = torch.arange(y_full.shape[1], dtype=torch.float).unsqueeze(-1)

        y_train, y_test, _, lab_test = train_test_split(y, label, test_size=test_ratio, random_state=self.seed)


        x_train = x_full.repeat(y_train.shape[0], 1, 1)
        y_train = torch.tensor(y_train).unsqueeze(-1)

        y_valid, y_test, lab_val, lab_test = train_test_split(y_test, lab_test, test_size=0.5, random_state=self.seed)

        x_valid = x_full.repeat(y_valid.shape[0], 1, 1)
        x_test = x_full.repeat(y_test.shape[0], 1, 1)

        y_valid = torch.tensor(y_valid).unsqueeze(-1)
        y_test = torch.tensor(y_test).unsqueeze(-1)


        y_train = self.scaler.fit_transform(y_train)
        y_valid = self.scaler.transform(y_valid)
        y_test = self.scaler.transform(y_test)

        return x_train, y_train, x_valid, y_valid, x_test, y_test
