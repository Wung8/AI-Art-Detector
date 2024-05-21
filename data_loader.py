

# Import the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import random
import glob
from PIL import Image
import numpy as np
from numpy import asarray
import torchvision.transforms.functional as fn



class training_data(torch.utils.data.Dataset):
  ##Characterizes a dataset for PyTorch
  def __init__(self):
        ##Initialization
        print('Getting Data')
        self.data = []
        ai_images_filenames = glob.glob('ai_images/*.png')
        artist_images_filenames = glob.glob('artist_images/*.png')

        l = len(ai_images_filenames)

        print()
        for i,img in enumerate(ai_images_filenames):
          if i/l//.1 > (i-1)/l//.1: print('.',end='')
          img = self.formatImg(img)
          label = torch.tensor([1,0]).type(torch.float)
          self.data.append([img, label])

        print()
        for i,img in enumerate(artist_images_filenames):
          if i/l//.1 > (i-1)/l//.1: print('.',end='')
          img = self.formatImg(img)
          label = torch.tensor([0,1]).type(torch.float)
          self.data.append([img, label])
          
        print(f'\n{len(self.data)} samples')

  def formatImg(self,img_path):
    img = Image.open(img_path)
    # converts img to np and removes transparency values
    img = asarray(img, dtype=np.float16)
    img = img[:,:,:3]
    # converts np to torch and makes the channel value the first one
    img = torch.from_numpy(img).type(torch.float)
    img = np.swapaxes(img,0,2)
    # resizes and crops for resnet
    img = fn.resize(img,size=[224])
    img = fn.center_crop(img,output_size=[224,224])
    img = img/255
    return img

  def __len__(self):
        ##Denotes the total number of samples
        return len(self.data)

  def __getitem__(self, index):
        ##Selects one sample of data
        return self.data[index]


class testing_data(torch.utils.data.Dataset):
  ##Characterizes a dataset for PyTorch
  def __init__(self,folder='test_images/*.png'):
        ##Initialization
        print('Getting Data')
        self.data = []
        filenames = glob.glob(folder)

        for i,img in enumerate(filenames[:100]):
          img = self.formatImg(img)
          label = torch.tensor([0,1]).type(torch.float)
          self.data.append([img, filenames[i]])
          
        print(f'{len(self.data)} samples')

  def formatImg(self,img_path):
    img = Image.open(img_path)
    # converts img to np and removes transparency values
    img = asarray(img, dtype=np.float16)
    img = img[:,:,:3]
    # converts np to torch and makes the channel value the first one
    img = torch.from_numpy(img).type(torch.float)
    img = np.swapaxes(img,0,2)
    # resizes and crops for resnet
    img = fn.resize(img,size=[224])
    img = fn.center_crop(img,output_size=[224,224])
    img = img/255
    return img

  def __len__(self):
        ##Denotes the total number of samples
        return len(self.data)

  def __getitem__(self, index):
        ##Selects one sample of data
        return self.data[index]


      
