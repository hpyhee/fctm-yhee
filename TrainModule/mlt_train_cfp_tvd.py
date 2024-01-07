# %%
import sys
sys.path.append("Towards-Realtime-MOT")
import os
import torch
from models import Darknet
from utils import datasets
from tqdm import tqdm
import torch
import math
from source.ipca import IPCA

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
yolo = Darknet("./Towards-Realtime-MOT/cfg/yolov3_1088x608.cfg", nID=14455)
yolo.load_state_dict(torch.load("./Towards-Realtime-MOT/weights/jde.1088x608.uncertainty.pt", map_location='cpu')['model'], strict=False)
yolo.cuda().eval()
seq_path = "./trainset/tvd/"

dataloader = datasets.LoadImages(seq_path)

d2 = torch.nn.PixelUnshuffle(2)
d4 = torch.nn.PixelUnshuffle(4)

n_components= int(7168*0.0625)
n_feature = int(7168)#DN53
ipca = IPCA(n_components, n_feature)
ipca = ipca.to("cuda")
for idx, (img_path, img, img0) in enumerate(tqdm(dataloader)):
    with torch.no_grad():
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        x = yolo.part1(blob, [36, 61,  74])#DN53
        X = torch.concat([d4(x[0]), d2(x[1]), x[2]], dim=1)#DN53
        A = X.reshape(( X.shape[1], X.shape[2]*X.shape[3]))
        A = A.transpose(0, 1)
        ipca.fit(A)
torch.save(ipca, f"mlt_track_dn53_{n_components}x{n_feature}_cfp.pt") #no weight