from torch import nn
import torch


class BpNet(nn.Module):
    def __init__(self, net_size):
        super(BpNet, self).__init__()
        # this is a two-layer fully-connected network
        self.net = nn.Sequential(nn.Linear(net_size[0], net_size[1]), \
                                 nn.ReLU(), \
                                 nn.Linear(net_size[1], net_size[2]), \
                                 nn.ReLU(), \
                                 nn.Linear(net_size[2], net_size[3]))

    def forward(self, x):
        return self.net(x)

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        ckpt = torch.load(path)
        self.net.load_state_dict(ckpt)
