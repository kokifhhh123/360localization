from pathlib import Path
from LightGlue.lightglue import LightGlue, SuperPoint, DISK
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d
import torch
import numpy as np
from module import MLP_module


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features='superpoint').eval().to(device)

torch.set_grad_enabled(False)
images = Path('/mnt/home_6T/public/weien/area1/')

image0 = load_image(images / '1.8.png')
image1 = load_image(images / '1.14.png')
feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))

# model = MLP_module().to(device)

PATH = '/mnt/home_6T/public/weien/MLP_checkpoint/model_20230925_214348_185'
model = MLP_module()
model.load_state_dict(torch.load(PATH))
model.eval()


# print(feats0['descriptors'].shape)
# print(feats0['descriptor_all'].shape)
# print(feats0['image_size'])
# matches01 = matcher({'image0': feats0, 'image1': feats1})
