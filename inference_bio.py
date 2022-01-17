# Python Standard Libraries
from datetime import datetime
import glob
import itertools
import math
import os
import random
import re
import time
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Deep Learning libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision

# Other libraries
# ~ Scientific
import numpy as np
import scipy.stats as st
# ~ Image manipulation / visualisation
import imgaug
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage
import skimage.io as skio
import skimage.transform as sktr
# ~ Other
from tqdm.notebook import tqdm

# Local libraries
from utils.image import *
from utils.plotting import *
from utils.torch import *

# IPython
from IPython.display import clear_output, HTML

import time
import sys
import os

os.system('mkdir -p ./results/export')

# Modality slicing
# You can choose a set of channels per modality (RGB for instance)
# Modality A
modA = slice(0, 1)
modA_name = "SHG"
modA_len = modA.stop - modA.start
# Modality B
modB = slice(1, 4)
modB_name = "BF"
modB_len = modB.stop - modB.start

from models.tiramisu import DenseUNet

class SlideDataset(Dataset):
    def __init__(self, folder_path, name_regex=r"(?P<name>.*_(?P<type>.*)\.", logSHG=True, transform=None):
        self.transform = transform
        if not isinstance(folder_path, list):
            folder_path = [folder_path]
        self.path = folder_path
        self.filenames = [glob.glob(path) for path in folder_path]
        self.filenames = list(itertools.chain(*self.filenames))

        dataset = {}
        #pbar = tqdm(total=len(self.filenames))
        for pathname in self.filenames:
            filename = os.path.basename(pathname)
            #pbar.set_description(filename)
            m = re.search(name_regex, filename, flags=re.IGNORECASE)
            assert m is not None, f"Couldn't find filename in {filename}."
            file_id = m.group("name")
            file_type = m.group("type")

            if file_id not in dataset.keys():
                dataset[file_id] = {}

            img = skio.imread(pathname)
            img = skimage.img_as_float(img)

            if file_type == "SHG" and logSHG:
                img = np.log(1.+img)

            if img.ndim == 2:
                img = img[..., np.newaxis]
            dataset[file_id][file_type] = img
            #pbar.update(1)

        print(dataset.keys())

        self.images = []
        for image_set in dataset:
            try:
                self.images.append(
                    np.block([
                        dataset[image_set]["SHG"],
                        dataset[image_set]["BF"]
                    ]).astype(np.float32)
                )
            except ValueError:
                print(f"Failed concatenating set {image_set}. Shapes are {dataset[image_set]['SHG'].shape} and {dataset[image_set]['BF'].shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx, augment=True):
        if augment and self.transform:
            return self.transform(self.images[idx])
        return self.images[idx]

class ImgAugTransform:
    def __init__(self, testing=False):
        if not testing:
            self.aug = iaa.Sequential([
                iaa.CropToFixedSize(128,128),
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-180, 180), order=[0, 1, 3], mode="symmetric"),
            ])
        else:
            self.aug = iaa.Sequential([
                iaa.CropToFixedSize(128,128),
            ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

class ModNet(DenseUNet):
    def __init__(self, **args):
        super(ModNet, self).__init__(**args, include_top=False)
        out_channels = self.get_channels_count()[-1]
        self.final_conv = torch.nn.Conv2d(out_channels, latent_channels, 1, bias=False)

    def forward(self, x):
        # Penultimate layer
        L_hat = super(ModNet, self).forward(x)
        # Final convolution
        return self.final_conv(L_hat)

print(len(sys.argv), sys.argv)
if len(sys.argv) < 3:
    print('Use: inference_bio.py model_path dataset_base_path')
    sys.exit(-1)

# Models avaible in the github release
checkpoint = torch.load(sys.argv[1])
#checkpoint = torch.load("models/model_biodata_cosine.pt")
modelA = checkpoint['modelA']
modelB = checkpoint['modelB']
device = "cpu"
modelA.to(device)
modelB.to(device)
modelA.eval()
modelB.eval()

# Number of threads to use
# It seems to be best at the number of physical cores when hyperthreading is enabled
# In our case: 18 physical + 18 logical cores
torch.set_num_threads(5)

dset_registration = SlideDataset([
    sys.argv[2] + "/TestSet/*_SHG.tif",
    sys.argv[2] + "/TestSet/*_BF.tif"
], name_regex=r"(?P<name>[a-z0-9_]+)_(?P<type>[A-Z]+)", transform=ImgAugTransform(testing=True))

# How many images to compute in one iteration?
batch_size = 1

all_paths = []

N = len(dset_registration)
l, r = 0, batch_size
idx = 1
for i in range(int(np.ceil(N / batch_size))):
    batch = []
    for j in range(l, r):
        batch.append(dset_registration.get(j, augment=False))
    batch = torch.tensor(np.stack(batch), device=device).permute(0, 3, 1, 2)

    newdim = (np.array(batch.shape[2:]) // 128) * 128
    L1 = modelA(batch[:, modA, :newdim[0], :newdim[1]])
    L2 = modelB(batch[:, modB, :newdim[0], :newdim[1]])
    
    for j in range(len(batch)):#L1.shape[0]):
        path1 = "/home/johan/CoMIR/results/export/" + os.path.splitext(os.path.basename(dset_registration.filenames[idx]))[0] + "_R1.tif"
        path2 = "/home/johan/CoMIR/results/export/" + os.path.splitext(os.path.basename(dset_registration.filenames[idx]))[0] + "_R2.tif"

        print(path1)
        print(path2)

        all_paths.append(path1)
        all_paths.append(path2)

        skio.imsave(path1, L1[j].permute(1, 2, 0).detach().numpy())
        skio.imsave(path2, L2[j].permute(1, 2, 0).detach().numpy())
        idx += 1

    l, r = l+batch_size, r+batch_size
    if r > N:
        r = N

all_paths = sorted(all_paths)
for i in range(len(all_paths)):
    print(all_paths[i])