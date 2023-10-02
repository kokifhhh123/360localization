from pathlib import Path
from LightGlue.lightglue import LightGlue, SuperPoint, DISK
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d
import torch
from torch import nn
import numpy as np
from torchrl.modules import MLP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
import itertools

def sigmoid_log_double_softmax(
        sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """ create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(
        sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)

    scores = sim.new_full((b, m+1, n+1), 0)
    scores[:, :m, :n] = (scores0 + scores1 + certainties)
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    exp = False
    if exp:
        scores_no = sim.new_full((b, m+1, n+1), 0)
        c_exp = F.sigmoid(z0) + F.sigmoid(z1).transpose(1, 2)
        s0_exp = F.softmax(sim, 2)
        s1_exp = F.softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
        scores_no[:, :m, :n] = (s0_exp + s1_exp + c_exp)
        scores_no[:, :-1, -1] = F.sigmoid(-z0.squeeze(-1))
        scores_no[:, -1, :-1] = F.sigmoid(-z1.squeeze(-1))
    else:
        scores_no = F.sigmoid(scores.clone())
    return scores, scores_no

class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """ build assignment matrix from descriptors """
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**.25, mdesc1 / d**.25
        sim = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores, scores_no = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim, scores_no

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)
 
class MLP_module(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 256
        n_layers = 1
        self.MLP = MLP(in_features=256, out_features=3, num_cells=[128, 64, 32, 16])
        self.log_assignment = nn.ModuleList(
            [MatchAssignment(dim) for _ in range(n_layers)])
        self.MLP_de = MLP(in_features=3, out_features=256, num_cells=[16, 32, 64, 128])


    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):

        desc0_mlp = self.MLP(desc0)
        desc1_mlp = self.MLP(desc1)
        # scores_mlp, _, scores_no = self.log_assignment[0](desc0_mlp, desc1_mlp)
        # desc0_back = self.MLP_de(desc0_mlp)
        # desc1_back = self.MLP_de(desc1_mlp)

        # scores_mlp, _, scores_no = self.log_assignment[0](desc0_back, desc1_back)
        return desc0_mlp,desc1_mlp
