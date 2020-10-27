import torch
from torch import nn

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
device = torch.device(device)

class DQN(nn.Module):
    def __init__(self, in_size = 3654, out_size = 23):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Linear(in_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(256, out_size)
        )

        self.to(device)

    def forward(self, x):
        x = x.float().to(device)

        x = self.initial(x)
        x = self.head(x)

        return x

