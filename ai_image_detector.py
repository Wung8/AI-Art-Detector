

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import glob
from PIL import Image
from numpy import asarray

from detector_network import Network
from data_loader import training_data


model =  models.resnet18(weights=True,progress=False)
# add a fc network to the end of resnet so it only has 2 outputs
model_ = Network()
model.fc = model_
print(model)

#loads training data
train_data = training_data()

test_size = 100
batch_size = 10

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
 )


optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

#model.load_state_dict(torch.load('model_10.pt',map_location=torch.device('cpu')))

model.to(device)

for epoch in range(15):
    print(f'starting epoch {epoch+1}')
    correct = 0
    for i, (images, labels) in enumerate(train_dataloader):
        images,labels = images.to(device),labels.to(device)
        
        # Run the forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels.squeeze())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if len(labels) < 10: continue
        for j in range(10):
            output_tolist = outputs.squeeze().tolist()
            if labels.tolist()[j].index(max(labels.tolist()[j])) == output_tolist[j].index(max(output_tolist[j])):
                correct += 1
        if (i + 1) % test_size == 0:
            print(f'epoch:{epoch + 1} batch:{i+1} | accuracy: {correct}/{batch_size*test_size} | {labels.tolist()[0].index(1.0)} {outputs.squeeze().tolist()[0].index(max(outputs.squeeze().tolist()[0]))}')
            correct = 0

    if epoch%5 == 0: torch.save(model.state_dict(), f'art-model_{epoch}.pt')


print('Finished Training')

torch.save(model.state_dict(), 'art-model_final.pt')
