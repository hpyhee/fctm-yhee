# %%
import sys
sys.path.append("detectron2")
sys.path.append("Towards-Realtime-MOT")
import os
import torch
from utils import datasets
from tqdm import tqdm
import torch
import math
from source.det2 import *
from source.ipca import IPCA
from source.mlt2 import *


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"


mlt     = MLT("", mode="train", device=DEVICE)
model_cfg_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_cfg_name))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg_name)
cfg.MODEL.DEVICE = "cpu"
task = "seg"
NNtask = DET2(cfg, task, "voc")    
# seq_path = "./trainset/detseg"
# dataloader = datasets.LoadImages(seq_path)
train_list = os.listdir("./trainset/detseg")
dataloader = mlt.get_image_loader(NNtask,path= "./trainset/detseg", img_list=train_list)
d2 = torch.nn.PixelUnshuffle(2)
d4 = torch.nn.PixelUnshuffle(4)
d8 = torch.nn.PixelUnshuffle(8)

n_components= int(256)
n_feature = int(21760)#ALT1
ipca = IPCA(n_components, n_feature)
ipca = ipca.to("cpu")

for path, img, img_size in tqdm(dataloader):
    print(path)
    # NN task part 1
    x = NNtask.part1(img)
    X = torch.concat([d8(x[0]), d4(x[1]), d2(x[2]), x[3]], dim=1)
    A = X.reshape(( X.shape[1], X.shape[2]*X.shape[3]))
    A = A.transpose(0, 1)
    ipca.fit(A)

torch.save(ipca, f"mlt_{task}_player_{n_components}x{n_feature}_cfp.pt")