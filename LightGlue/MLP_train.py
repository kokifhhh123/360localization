from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
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
        dim = 3
        n_layers = 1
        self.MLP = MLP(in_features=256, out_features=3, num_cells=[128, 64, 32, 16])
        self.log_assignment = nn.ModuleList(
            [MatchAssignment(dim) for _ in range(n_layers)])
        self.MLP_de = MLP(in_features=3, out_features=256, num_cells=[16, 32, 64, 128])
    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):

        desc0_mlp = self.MLP(desc0)
        desc1_mlp = self.MLP(desc1)
        scores_mlp, _, scores_no = self.log_assignment[0](desc0_mlp, desc1_mlp)
        desc0_back = self.MLP_de(desc0_mlp)
        desc1_back = self.MLP_de(desc1_mlp)
        return scores_no, desc0_back, desc1_back

class MLPDataset(Dataset):

    # data loading
    def __init__(self):
        self.data_root = './tmp5/'
        # self.data_path = '/mnt/home_6T/public/weien/MLP_data/room'+str(index)+'.pt'
        nums = [str(i) for i in range(1, 21)]
        r = 2
        combinations = list(itertools.combinations(nums, r))
        self.data_comb = []
        for i in range(1, 16):
            if i==4:
                continue
            for j in combinations:
                self.data_comb.append([str(i), j[0], j[1]])
        # self.data = torch.load(self.data_path)
    # working for indexing
    def __getitem__(self, index):
        img0 = load_image(self.data_root+self.data_comb[index][0]+'.'+self.data_comb[index][1]+'.png')
        img1 = load_image(self.data_root+self.data_comb[index][0]+'.'+self.data_comb[index][2]+'.png')
        
        return img0, img1
    # return the length of our dataset
    def __len__(self):
        return len(self.data_comb)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = MLP_module().to(device)
loss_fn = nn.MSELoss()
#loss_L1 = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features='superpoint').eval().to(device)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        img0, img1 = data
        with torch.no_grad():
            feats0 = extractor.extract(img0.clone().detach().to(device))
            feats1 = extractor.extract(img1.clone().detach().to(device))
            matches01 = matcher({'image0': feats0, 'image1': feats1})
            input0, input1, label = matches01['input0'], matches01['input1'], matches01['label']
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs, d0_back, d1_back = model(input0, input1)
        
        
        # Compute the loss and its gradients
        loss = loss_fn(outputs, label)
        loss_d0 = loss_fn(d0_back, input0)
        loss_d1 = loss_fn(d1_back, input1)
        loss_total = 0.7*loss + loss_d0 + loss_d1
        #with torch.autograd.detect_anomaly():
        loss_total.backward()

        # gradient clipping
        #torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
        
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss_total.item()
        #print(loss.item())
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            #tb_x = epoch_index * len(train_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('/mnt/home_6T/public/weien/MLP_checkpoint/runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 500

best_vloss = 1_000_000.
seed = 100
torch.manual_seed(seed)
dataset = MLPDataset()
train_set, val_set = random_split(dataset, [0.7, 0.3])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True)
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vimg0, vimg1 = vdata
            vfeats0 = extractor.extract(vimg0.to(device))
            vfeats1 = extractor.extract(vimg1.to(device))
            vmatches01 = matcher({'image0': vfeats0, 'image1': vfeats1})
            vin0, vin1, vlabels = vmatches01['input0'], vmatches01['input1'], vmatches01['label']
            voutputs, vd0_back, vd1_back = model(vin0, vin1)
            vloss = loss_fn(voutputs, vlabels)
            vloss_d0 = loss_fn(vd0_back, vin0)
            vloss_d1 = loss_fn(vd1_back, vin1)
            vloss_total = 0.7*vloss + vloss_d0 + vloss_d1
            running_vloss += vloss_total

    avg_vloss = running_vloss / (i + 1)
    scheduler.step(avg_vloss)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    #writer.add_scalar('Loss/train', avg_loss, epoch_number+1)
    #writer.add_scalar('Loss/validation', avg_vloss, epoch_number+1)
    writer.add_scalars('Training vs. Validation',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number+1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = '/mnt/home_6T/public/weien/MLP_checkpoint/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1