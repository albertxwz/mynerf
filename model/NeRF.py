import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(
            self,
            hidden_size: int = 256,
            L_embed_loc: int = 10,
            L_embed_dir: int = 4,
            use_dir: bool = True,
        ) -> None:
        super(NeRF, self).__init__()

        input_size = 3 + 3 * 2 * L_embed_loc

        self.use_dir = use_dir
        
        dir_size = 0
        if use_dir:
            dir_size = 3 + 3 * 2 * L_embed_dir


        self.block1 = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_size + input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )

        self.output_sigma = nn.Sequential(
            nn.Linear(hidden_size, 1), nn.ReLU(),
        )

        self.output_rgb = nn.Sequential(
            nn.Linear(hidden_size + dir_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 3), nn.Sigmoid(),
        )
    
    def forward(self, x, dirs=None): # x: N_rays x embed_size
        out = self.block1(x)
        out = self.block2(torch.concat([out, x], -1))
        sigma = self.output_sigma(out)

        if self.use_dir:
            out = torch.concat([out, dirs], -1)

        out = self.output_rgb(out)

        return out, sigma

# test
# model = NeRF()

# x = torch.rand(10, 63)
# dir = torch.rand(10, 27)

# rgb, sigma = model(x, dir)

# print("shape: [rgb, sigma]=", rgb.shape, sigma.shape)