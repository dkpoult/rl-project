import torch
from torch import nn

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
device = torch.device(device)

class BottleneckModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.bottle = nn.Sequential(
            # nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, padding = 1),
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 1),
            nn.ReLU(),
            # nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.float().to(device)

        residual = x
        x = self.bottle(x)

        return x + residual


class DQN(nn.Module):
    def __init__(self, out_size = 23):
        super().__init__()

        self.map_branch_root = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 128, 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.ReLU()
        )

        self.bottle1 = BottleneckModule()
        self.pooling1 = nn.MaxPool2d(2)

        self.map_branch_head = nn.Sequential(
            nn.Conv2d(256, 32, 3, padding = 1),
            nn.Flatten(),
            nn.Linear(12480, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.stats_branch = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            # nn.LSTM(288, 256, 2),
            nn.Linear(288, out_size)
        )

        self.to(device)

    def forward(self, x):
        maps = torch.stack([inputs[0].float() for inputs in x]).to(device)
        stats = torch.stack([inputs[1].float() for inputs in x]).to(device)

        maps_x = self.map_branch_root(maps)
        # maps_x = self.bottle1(maps_x)
        maps_x = self.pooling1(maps_x)
        maps_x = self.map_branch_head(maps_x)

        stats_x = self.stats_branch(stats)

        x = torch.cat([maps_x, stats_x], dim = -1)
        x = self.head(x)

        return x

