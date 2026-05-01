import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self):
        """
        Q Network - NN to compute Q(s,a) values
        Target network == Q network with frozen parameters per some C steps

        For cartpole, simple MLP:
        - 4 inputs: cart position, cart vel, pole angle, pole angular vel
        - 2 hidden layers, each of size 64
        - 2 outputs: Q value for each of the possible actions (push cart left, push cart right)
        - ReLU activations
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        q_values = self.mlp(x)
        return q_values