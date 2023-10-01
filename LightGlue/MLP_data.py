from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import numpy as np
import os
import itertools
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features='superpoint').eval().to(device)

data_path = '/mnt/home_6T/public/weien/tmp5/'
images = os.listdir(data_path)
data = []
nums = [str(i) for i in range(1, 21)]
r = 2
combinations = list(itertools.combinations(nums, r))
for i in range(1, 16):
    if i==4:
        continue
    for j in combinations:
        image0 = load_image(data_path+str(i)+'.'+j[0]+'.png')
        image1 = load_image(data_path+str(i)+'.'+j[1]+'.png')
        print(str(i)+'.'+j[0]+'.png'+'/'+str(i)+'.'+j[1]+'.png')
        feats0 = extractor.extract(image0.to(device))
        feats1 = extractor.extract(image1.to(device))
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        data_tmp = [matches01['input0'].squeeze(0), matches01['input1'].squeeze(0), matches01['label'].squeeze(0)]
        data.append(data_tmp)
    torch.save(data, '/mnt/home_6T/public/weien/MLP_data/room'+str(i)+'.pt')
    print('room'+str(i)+'.pt')
    data = []

