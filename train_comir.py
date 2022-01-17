
# Python Standard Libraries
from datetime import datetime
import glob
import itertools
import math
import os
import sys
import random
import time
import warnings

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
import skimage
import skimage.io as skio
import skimage.transform as sktr

# Local libraries
from utils.image import *
from utils.torch import *

from models.tiramisu import DenseUNet

count = torch.cuda.device_count()
print(f"{count} GPU device(s) available.")
print()
print("List of GPUs:")
for i in range(count):
    print(f"* {torch.cuda.get_device_name(i)}")

def read_args():
    pths = {}
    if len(sys.argv) >= 3:
        pths['train_path_a'] = sys.argv[1]
        pths['train_path_b'] = sys.argv[2]
    else:
        pths['train_path_a'] = './data/mod_a'
        pths['train_path_b'] = './data/mod_b'
        print('No training set provided. Using default path ./data/mod_a and ./data/mod_b')

    if len(sys.argv) >= 5:
        pths['val_path_a'] = sys.argv[3]
        pths['val_path_b'] = sys.argv[4]
    else:
        pths['val_path_a'] = None
        pths['val_path_b'] = None
        print('No validation set provided.')
    return pths

pths = read_args()

# DATA RELATED
modA_train_path = pths['train_path_a']#'/data/johan/biodata/TrainSet2/SHG/*'
modB_train_path = pths['train_path_b']#'/data/johan/biodata/TrainSet2/BF/*'
modA_val_path = pths['val_path_a']#'/data/johan/biodata/Validation1Set/SHG/*'
modB_val_path = pths['val_path_b']#'/data/johan/biodata/Validation1Set/BF/*'

# METHOD RELATED
# The place where the models will be saved
export_folder = "results" # Add this path to the .gitignore
# The number of channels in the latent space
latent_channels = 1

logTransformA = False
logTransformB = False

# Distance function
simfunctions = {
    "euclidean" : lambda x, y: -torch.norm(x - y, p=2, dim=1).mean(),
    "L1"        : lambda x, y: -torch.norm(x - y, p=1, dim=1).mean(),
    "MSE"       : lambda x, y: -(x - y).pow(2).mean(),
    "MPE15"       : lambda x, y: -(torch.abs(x - y)).pow(1.5).mean(),
    "MPE25"       : lambda x, y: -(torch.abs(x - y)).pow(2.5).mean(),
    "L3"        : lambda x, y: -torch.norm(x - y, p=3, dim=1).mean(),
    "Linf"      : lambda x, y: -torch.norm(x - y, p=float("inf"), dim=1).mean(),
    "soft_corr" : lambda x, y: F.softplus(x*y).sum(axis=1),
    "corr"      : lambda x, y: (x*y).sum(axis=1),
    "cosine"    : lambda x, y: F.cosine_similarity(x, y, dim=1, eps=1e-8).mean(),
    "angular"   : lambda x, y: F.cosine_similarity(x, y, dim=1, eps=1e-8).acos().mean() / math.pi,
}
sim_func = simfunctions["MSE"]
# Temperature (tau) of the loss
tau = 0.5
# L1/L2 activation regularization
act_l1 = 1e-4 # settings for MSE, MSElong, MSEverylong
act_l2 = 1e-4 # settings for MSE, MSElong, MSEverylong
# p4 Equivariance
equivariance = True

# DEEP LEARNING RELATED
# Device to train on (inference is done on cpu)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use two GPUs?
device1 = device2 = device # 1 gpu for 2 modalities
#device1, device2 = "cuda:0", "cuda:1" # 1 gpu per modality
# Arguments for the tiramisu neural network
tiramisu_args = {
    # Number of convolutional filters for the first convolution
    "init_conv_filters": 32,
    # Number and depth of down blocks
    "down_blocks": (4, 4, 4, 4, 4, 4),
    # Number and depth of up blocks
    "up_blocks": (4, 4, 4, 4, 4, 4),
    # Number of dense layers in the bottleneck
    "bottleneck_layers": 4,
    # Upsampling type of layer (upsample has no grid artefacts)
    "upsampling_type": "upsample",
    # Type of max pooling, blurpool has better shift-invariance
    "transition_pooling": "max",
    # Dropout rate for the convolution
    "dropout_rate": 0.0,#0.2
    # Early maxpooling to reduce the input size
    "early_transition": False,
    # Activation function at the last layer
    "activation_func": None,
    # How much the conv layers should be compressed? (Memory saving)
    "compression": 0.75,
    # Memory efficient version of the tiramisu network (trades memory for computes)
    # Gains of memory are enormous compared to the speed decrease.
    # See: https://arxiv.org/pdf/1707.06990.pdf
    "efficient": True,
}
# Epochs
epochs = 100 #23
# Batch size
modifier = 4
batch_size = 32//modifier
# Steps per epoch
steps_per_epoch = 32*modifier
# Number of steps
steps = steps_per_epoch * epochs
# How many unique patches are fed during one epoch
samples_per_epoch = steps_per_epoch * batch_size
# Specifies the patch size to train on
patch_sz = 128

num_workers = 4
# Optimiser
#from lars.lars import LARS
#optimiser = LARS
optimiser = optim.SGD
# Optimizer arguments
opt_args = {
    "lr": 1e-2,
    "weight_decay": 1e-5,
    "momentum": 0.9
}
# Gradient norm. (limit on how big the gradients can get)
grad_norm = 1.0

# DATASET RELATED
def worker_init_fn(worker_id):
    base_seed = int(torch.randint(2**32, (1,)).item())
    lib_seed = (base_seed + worker_id) % (2**32)
    imgaug.seed(lib_seed)
    np.random.seed(lib_seed)

dataloader_args = {
    "batch_size": batch_size,
    "shuffle": False,
    "num_workers": num_workers,
    "pin_memory": True,
    "worker_init_fn": worker_init_fn,
}

# Create 

if not os.path.exists(export_folder):
    os.makedirs(export_folder)
    print("Created export folder!")

def filenames_to_dict(filenamesA, filenamesB):
    d = {}
    for i in range(len(filenamesA)):
        basename = os.path.basename(filenamesA[i])
        d[basename] = (i, None)
    for i in range(len(filenamesB)):
        basename = os.path.basename(filenamesB[i])
        # filter out files only in B
        if basename in d:
            d[basename] = (d[basename][0], i)

    # filter out files only in A
    d = {k:v for k,v in d.items() if v[1] is not None}
    return d

class MultimodalDataset(Dataset):
    def __init__(self, pathA, pathB, logA=False, logB=False, transform=None):
        self.transform = transform

        if not isinstance(pathA, list):
            pathA = [pathA]
        if not isinstance(pathB, list):
            pathB = [pathB]
        self.pathA = pathA
        self.pathB = pathB
        self.filenamesA = [glob.glob(path) for path in pathA]
        self.filenamesA = list(itertools.chain(*self.filenamesA))
        self.filenamesB = [glob.glob(path) for path in pathB]
        self.filenamesB = list(itertools.chain(*self.filenamesB))

        self.channels = [None, None]

        filename_index_pairs = filenames_to_dict(self.filenamesA, self.filenamesB)
        
        filenames = [self.filenamesA, self.filenamesB]
        log_flags = [logA, logB]

        dataset = {}
        for mod_ind in range(2):
            # Read all files from modality
            for filename, inds in filename_index_pairs.items():
                pathname = filenames[mod_ind][inds[mod_ind]]

                filename = os.path.basename(pathname)
                
                if filename not in dataset.keys():
                    dataset[filename] = [None, None]

                img = skio.imread(pathname)
                img = skimage.img_as_float(img)

                if log_flags[mod_ind]:
                    img = np.log(1.+img)

                if img.ndim == 2:
                    img = img[..., np.newaxis]

                if self.channels[mod_ind] is None:
                    self.channels[mod_ind] = img.shape[2]

                dataset[filename][mod_ind] = img

        self.images = []
        for image_set in dataset:
            try:
                self.images.append(
                    np.block([
                        dataset[image_set][0],
                        dataset[image_set][1]
                    ]).astype(np.float32)
                )
            except ValueError:
                print(f"Failed concatenating set {image_set}. Shapes are {dataset[image_set][0].shape} and {dataset[image_set][1].shape}")

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
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-180, 180), order=[0, 1, 3], mode="symmetric"),
                iaa.Sometimes(
                    0.5,
                    iaa.GaussianBlur(sigma=(0, 2.0))),
                iaa.Sometimes(
                    0.5,
                    iaa.AdditiveGaussianNoise(loc=0.0, scale=(0.0, 0.1), per_channel=1.0)),
                iaa.Sometimes(
                    0.5,
                    iaa.LinearContrast((0.8, 1.0/0.8), per_channel=0.5)),
                iaa.CropToFixedSize(patch_sz,patch_sz,position='uniform',seed=1337),
            ])
        else:
            self.aug = iaa.Sequential([
                iaa.CropToFixedSize(patch_sz,patch_sz,position='uniform',seed=1337),
            ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

print("Loading train set...")
dset = MultimodalDataset(modA_train_path + '/*', modB_train_path + '/*', logA=logTransformA, logB=logTransformB, transform=ImgAugTransform())
if modA_val_path is not None and modB_val_path is not None:
    validation_enabled = True
    print("Loading test set...")
    dset_test = MultimodalDataset(modA_val_path + '/*', modB_val_path + '/*', logA=logTransformA, logB=logTransformB, transform=ImgAugTransform(testing=True))

# Modality slicing
# You can choose a set of channels per modality (RGB for instance)
# Modality A
modA_len = dset.channels[0]
modA = slice(0, modA_len)
modA_name = "A"
# Modality B
modB_len = dset.channels[1]
modB = slice(modA_len, modA_len + modB_len)
modB_name = "B"
print('Modality A has ', modA_len, ' channels.', sep='')
print('Modality B has ', modB_len, ' channels.', sep='')

train_loader = torch.utils.data.DataLoader(
    dset,
    sampler=OverSampler(dset, samples_per_epoch),
    **dataloader_args
)
if validation_enabled:
    test_loader = torch.utils.data.DataLoader(
        dset_test,
        sampler=OverSampler(dset_test, samples_per_epoch),
        **dataloader_args
    )

# Create model

class ModNet(DenseUNet):
    def __init__(self, **args):
        super(ModNet, self).__init__(**args, include_top=False)
        out_channels = self.get_channels_count()[-1]
        self.final_conv = torch.nn.Conv2d(out_channels, latent_channels, 1, bias=False)
        # This is merely for the benefit of the serialization (so it will be known in the inference)
        self.log_transform = False

    def set_log_transform(self, flag):
        # This is merely for the benefit of the serialization (so it will be known in the inference)
        self.log_transform = flag

    def forward(self, x):
        # Penultimate layer
        L_hat = super(ModNet, self).forward(x)
        # Final convolution
        return self.final_conv(L_hat)

torch.manual_seed(0)
modelA = ModNet(in_channels=modA_len, nb_classes=latent_channels, **tiramisu_args).to(device1)
modelB = ModNet(in_channels=modB_len, nb_classes=latent_channels, **tiramisu_args).to(device2)

# This is merely for the benefit of the serialization (so it will be known in the inference)
modelA.set_log_transform(logTransformA)
modelB.set_log_transform(logTransformB)

optimizerA = optimiser(modelA.parameters(), **opt_args)
optimizerB = optimiser(modelB.parameters(), **opt_args)

print("*** MODEL A ***")
modelA.summary()

modelA = modelA.to(device1)
modelB = modelB.to(device2)
torch.manual_seed(0)

def compute_pairwise_loss(Ls, similarity_fn, tau=1.0, device=None):
    """Computation of the final loss.
    
    Args:
        Ls (list): the latent spaces.
        similarity_fn (func): the similarity function between two datapoints x and y.
        tau (float): the temperature to apply to the similarities.
        device (str): the torch device to store the data and perform the computations.
    
    Returns (list of float):
        softmaxes: the loss for each positive sample (length=2N, with N=batch size).
        similarities: the similarity matrix with all pairwise similarities (2N, 2N)

    Note:
        This implementation works in the case where only 2 modalities are of
        interest (M=2). Please refer to the paper for the full algorithm.
    """
    # Computation of the similarity matrix
    # The matrix contains the pairwise similarities between each sample of the full batch
    # and each modalities.
    points = torch.cat([L.to(device) for L in Ls])
    N = batch_size
    similarities = torch.zeros(2*N, 2*N).to(device)
    for i in range(2*N):
        for j in range(i+1):
            s = similarity_fn(points[i], points[j])/tau
            similarities[i, j] = s
            similarities[j, i] = s

    # Computation of the loss, one row after the other.
    irange = np.arange(2*N)
    softmaxes = torch.empty(2*N).to(device)
    for i in range(2*N):
        j = (i + N) % (2 * N)
        pos = similarities[i, j]
        # The negative examples are all the remaining points
        # excluding self-similarity
        neg = similarities[i][irange != i]
        softmaxes[i] = -pos + torch.logsumexp(neg, dim=0)
    return softmaxes, similarities

def std_dev_of_loss(losses):
    if len(losses) < 2:
        return 0
    else:
        return np.std(losses, ddof=1)

def pos_error(similarities):
    N = batch_size
    sim_cpu = similarities.cpu()
    acc = 0
    for i in range(2*N):
        j = (i + N) % (2 * N)
        value = -sim_cpu[i, j]
        acc += value.item()
    return tau * acc / (2*N)

losses = {"train": [], "test": []}

def test(obj_func_size):
    """Runs the model on the test data."""
    obj_func_start = patch_sz // 2 - obj_func_sz // 2
    obj_func_end = obj_func_start + obj_func_sz    
    modelA.eval()
    modelB.eval()
    test_loss = []
    errors = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.permute(0, 3, 1, 2)
            dataA = data[:, modA].float().to(device1)
            dataB = data[:, modB].float().to(device2)
            L1 = modelA(dataA)
            L2 = modelB(dataB)

            L1 = L1[..., obj_func_start:obj_func_end, obj_func_start:obj_func_end]
            L2 = L2[..., obj_func_start:obj_func_end, obj_func_start:obj_func_end]

            softmaxes, similarities = compute_pairwise_loss(
                [L1, L2],
                similarity_fn=sim_func,
                tau=tau,
                device=device1
            )
            loss_test = softmaxes.mean()

            err = pos_error(similarities)
            errors.append(err)
            if act_l1 > 0.:
                loss_test += act_l1 * activation_decay([L1, L2], p=1, device=device1)
            if act_l2 > 0.:
                loss_test += act_l2 * activation_decay([L1, L2], p=2, device=device1)
            test_loss.append(loss_test.item())
            batch_progress = '[Batch:' + str(batch_idx+1) + '/' + str(steps_per_epoch) + ']'
            print('\r', batch_progress, ' Validation Loss: ', np.mean(test_loss), ' +- ', std_dev_of_loss(test_loss), ' (', np.mean(errors), ')   ', sep='', end='')
    losses["test"].append(np.mean(test_loss))
    
    print()

    return loss_test, similarities

obj_func_start = 0
obj_func_end = patch_sz
obj_func_sz = patch_sz
epoch = 0
for epoch in range(1, epochs+1):
    modelA.train()
    modelB.train()
    train_loss = []
    errors = []
    for batch_idx, data in enumerate(train_loader):
        # Preparing the batch
        data = data.permute(0, 3, 1, 2)
        dataA = data[:, modA].float().to(device1)
        dataB = data[:, modB].float().to(device2)
        # Reseting the optimizer (gradients set to zero)
        optimizerA.zero_grad()
        optimizerB.zero_grad()

        if equivariance:
            # Applies random 90 degrees rotations to the data (group p4)
            # This step enforces the formula of equivariance: d(f(T(x)), T^{-1}(f(x)))
            # With f(x) the neural network, T(x) a transformation, T^{-1}(x) the inverse transformation
            random_rotA = np.random.randint(4, size=batch_size)
            random_rotB = np.random.randint(4, size=batch_size)
            dataA_p4 = batch_rotate_p4(dataA, random_rotA, device1)
            dataB_p4 = batch_rotate_p4(dataB, random_rotB, device2)

            # Compute the forward pass
            L1 = modelA(dataA_p4)
            L2 = modelB(dataB_p4)

            # Applies the inverse of the 90 degree rotation to recover the right positions
            L1_ungrouped = batch_rotate_p4(L1, -random_rotA, device1)
            L2_ungrouped = batch_rotate_p4(L2, -random_rotB, device2)
        else:
            L1_ungrouped = modelA(dataA)
            L2_ungrouped = modelB(dataB)

        L1_ungrouped = L1_ungrouped[..., obj_func_start:obj_func_end, obj_func_start:obj_func_end]
        L2_ungrouped = L2_ungrouped[..., obj_func_start:obj_func_end, obj_func_start:obj_func_end]

        # Computes the loss
        softmaxes, similarities = compute_pairwise_loss(
            [L1_ungrouped, L2_ungrouped],
            similarity_fn=sim_func,
            tau=tau,
            device=device1
        )

        loss = softmaxes.mean()

        err = pos_error(similarities)

        # Activation regularization
        if act_l1 > 0.:
            loss += act_l1 * activation_decay([L1_ungrouped, L2_ungrouped], p=1., device=device1)
        if act_l2 > 0.:
            loss += act_l2 * activation_decay([L1_ungrouped, L2_ungrouped], p=2., device=device1)

        # Computing the gradients
        loss.backward()

        # Clipping the the gradients if they are too big
        torch.nn.utils.clip_grad_norm_(modelA.parameters(), grad_norm)
        torch.nn.utils.clip_grad_norm_(modelB.parameters(), grad_norm)
        # Performing the gradient descent
        optimizerA.step()
        optimizerB.step()

        train_loss.append(loss.item())
        errors.append(err)
        losses["train"].append(train_loss[-1])
        epoch_progress = '[Epoch:' + str(epoch) + '/' + str(epochs) + ']'
        batch_progress = '[Batch:' + str(batch_idx+1) + '/' + str(steps_per_epoch) + ']'
        print('\r', epoch_progress, batch_progress, ' Loss: ', np.mean(train_loss), ' +- ', std_dev_of_loss(train_loss), ' (', np.mean(errors), ')   ', sep='', end='')

    print()
    # Testing after each epoch
    if validation_enabled:
        _, similarities = test(obj_func_sz)

# Save model

date = datetime.now().strftime("%Y%d%m_%H%M%S")
model_path = os.path.join(export_folder, f"model_L{latent_channels}_{date}.pt")
latest_model_path = os.path.join(export_folder, f"latest.pt")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    torch.save({
        "modelA": modelA,
        "modelB": modelB,
    }, model_path)
    torch.save({
        "modelA": modelA,
        "modelB": modelB,
    }, latest_model_path)
print(f"model saved as: {model_path} and as {latest_model_path}")



