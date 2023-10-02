from pathlib import Path
from LightGlue.lightglue import LightGlue, SuperPoint, DISK
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d
import torch
import numpy as np
from module import MLP_module
import torchvision
import torchvision.transforms as T
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device) 
#  # load the extractor
# matcher = LightGlue(features='superpoint').eval().to(device)


torch.set_grad_enabled(False)
images = Path('/mnt/home_6T/public/weien/area1/')

path = '/mnt/home_6T/public/koki/gibson_tiny/Collierville/pano/'
img_path1 = path + 'rgb/point_p' + '000024' + '_view_equirectangular_domain_rgb.png'
img_path2 = path + 'rgb/point_p' + '000025' + '_view_equirectangular_domain_rgb.png'


image0 = load_image(img_path1)
# image0.resize((1024,512))
image1 = load_image(img_path2)

feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))

# model = MLP_module().to(device)

PATH = '/mnt/home_6T/public/weien/MLP_checkpoint/model_20230925_214348_185'
model = MLP_module().to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

print(feats0['descriptor_all'].shape)
print(feats1['descriptor_all'].shape)

mlp0,mlp1 = model(feats0['descriptor_all'],feats1['descriptor_all'])



# mlp0_numpy = mlp0.nump
print(mlp0.shape)
print(mlp1.shape)

m0 = mlp0.reshape((1,2048,1024,3))
m1 = mlp1.reshape((1,2048,1024,3))

print(m0.shape)
print(m1.shape)

transform = T.ToPILImage()

m0_image = transform(m0.squeeze(0).permute(2,1,0))
m1_image = transform(m0.squeeze(0).permute(2,1,0))
# m0_image.show()

m0_image.save('m0_image_1.png')
m1_image.save('m1_image_1.png')
# img = transform(tensor)
# print(feats0['descriptors'].shape)
# print(feats0['descriptor_all'].shape)
# print(feats0['image_size'])
# matches01 = matcher({'image0': feats0, 'image1': feats1})
