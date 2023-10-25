import torch

class QuadraticDataset(torch.utils.data.Dataset):
    def __init__(self, num_data, num_points, seed=42):
        super().__init__()

        self.num_data = num_data
        self.num_points = num_points
        self.seed = seed
        self.x = torch.linspace(start=-10., end=10., steps=self.num_points).unsqueeze(0).repeat(self.num_data, 1)
        self.dataset = self._create_dataset()

    def _create_dataset(self):
        torch.manual_seed(self.seed)
        a = torch.randint(low=0, high=2, size=(self.x.shape[0], 1)).repeat(1, self.num_points) * 2 - 1
        eps = torch.normal(mean=0., std=1., size=(self.x.shape[0], 1)).repeat(1, self.num_points)

        y = a * (self.x ** 2) + eps
        return y

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):

        return self.x[idx, :].unsqueeze(-1), self.dataset[idx, :].unsqueeze(-1) / 50
