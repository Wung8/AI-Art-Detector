import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision import models
from torch.utils.data import DataLoader
from data_loader import testing_data
import numpy as np
from PIL import Image

device = 'cpu'


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.fc1 = nn.Linear(512, 120)
        self.fc2 = nn.Linear(120, 2)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out
    

model =  models.resnet18(pretrained=True)
model_ = Network()
model.fc = model_
model.load_state_dict(torch.load('art-model_final.pt',map_location=torch.device('cpu')))
model.eval()


def formatImg(img_path):
    img = Image.open(img_path)
    # converts img to np and removes transparency values
    img = np.asarray(img, dtype=np.float16)
    img = img[:,:,:3]
    # converts np to torch and makes the channel value the first one
    img = torch.from_numpy(img).type(torch.float)
    img = np.swapaxes(img,0,2)
    # resizes and crops for resnet
    img = fn.resize(img,size=[224])
    img = fn.center_crop(img,output_size=[224,224])
    img = img/255
    img = img.unsqueeze(0)

    return img

def infer(img_path):
    img = formatImg(img_path)
    return model(img)
