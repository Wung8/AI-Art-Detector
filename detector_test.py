import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import os
import glob
from PIL import Image
from numpy import asarray

from detector_network import Network
from data_loader import testing_data


to_convert = glob.glob('test_images/*.jpg')

for file in to_convert:
  img = Image.open(file)
  img.save(file.replace('.jpg','.png'))
  os.remove(file)  


folder = 'test_images/*.png'

filenames = glob.glob(folder)
#filenames = glob.glob('test_images/*.png')


model =  models.resnet18(pretrained=True)
# add a fc network to the end of resnet so it only has 2 outputs
model_ = Network()
model.fc = model_

#loads training data
test_data = testing_data(folder)

test_size = 1
batch_size = 2

dataloader = DataLoader(
    test_data,
    batch_size=len(filenames),
    shuffle=False,
    num_workers=0,
 )


optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

model.load_state_dict(torch.load('art-model_final.pt',map_location=torch.device('cpu')))

A = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(dataloader):
    #for img,file in data:
      output = model(images)
      output_tolist = output.squeeze().tolist()
      label = output_tolist.index(max(output_tolist))
      zipped = zip(labels,output.tolist())
      #print(label,file)
      A = images

for val in zip(filenames,output.tolist()):
  print(val)



