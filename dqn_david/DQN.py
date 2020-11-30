import torch
from torch import nn

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
device = torch.device(device)

class Hourglass(nn.Module):
    def __init__(self):
        super().__init__()

        # Generic layers that can be used repeatedly
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor = 2)

        # First downsample convolution
        self.down_a = nn.Conv2d(128, 256, 3, padding = 1)
        # Second downsample convolution 
        self.down_b = nn.Conv2d(256, 512, 3, padding = 1)

        # Middle section to remap features
        self.mid = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding = 1),
            nn.Conv2d(1024, 512, 1),
        )

        # First upsampling convolution
        self.up_b = nn.Conv2d(512, 256, 3, padding = 1)
        # Second upsampling convolution
        self.up_a = nn.Conv2d(256, 128, 3, padding = 1)

        # One-by-one to apply after final upsampling
        self.final = nn.Conv2d(128, 128, 1)

    def forward(self, x):
        # Keep the initial as a residual
        residual_a = x

        # Convolve and downsample
        x = self.down_a(x)
        x = self.max_pool(self.relu(x))

        # Keep the second layer residual
        residual_b = x

        # Convolve and downsample again
        x = self.down_b(x)
        x = self.max_pool(self.relu(x))

        # Do the middle remapping
        x = self.mid(x)

        # Convolve, merge residual, and upsample
        x = self.up_b(x)
        x = self.upsample(self.relu(x) + residual_b[:,:,:x.shape[2],:x.shape[3]])

        # Convolve, merge residual, and upsample
        x = self.up_a(x)
        x = self.upsample(self.relu(x) + residual_a[:,:,:x.shape[2],:x.shape[3]])

        # Get our final result
        x = self.final(x)

        return x


class DQN(nn.Module):
    def __init__(self, out_size = 23):
        super().__init__()
        
        # Try embed the glyphs (optimistic)
        self.map_embedding = nn.Conv2d(1, 32, 1)

        # The root of the branch mapping
        self.map_branch_root = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding = 1),
            Hourglass(),
        )

        # Pooling to make the map manageable
        self.pooling1 = nn.MaxPool2d(2)

        # The head to give map features
        self.map_branch_head = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding = 1),
            nn.Flatten(),
            nn.Linear(12160, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        # The basic ANN for stats
        self.stats_branch = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Flatten(),
        )

        # The final ANN that'll give values
        self.head = nn.Sequential(
            nn.Linear(288, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_size),
            nn.ReLU()
        )

        self.to(device)

    def forward(self, x):
        # Separate our features
        maps_x, stats_x = x

        # "Embed" the map glyphs
        maps_x = self.map_embedding(maps_x)
        # Put them through the hourglass
        maps_x = self.map_branch_root(maps_x)
        # Shrink it because we don't have a 3090 TI
        maps_x = self.pooling1(maps_x)
        # Flatten and process it
        maps_x = self.map_branch_head(maps_x)

        # Process the stats
        stats_x = self.stats_branch(stats_x)

        # Combine our map output and stats output
        x = torch.cat([maps_x, stats_x], dim = -1)
        # Get the final result!
        x = self.head(x)

        # Give the values
        return x

