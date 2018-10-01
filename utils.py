#####################################
# METU - CENG483 HW3                #
# Author:                           #
#  Necip Fazil Yildiran             #
#####################################

from os import sep
from skimage import io, color
from skimage.transform import rescale
from skimage.viewer import ImageViewer
import numpy as np
import torch
import torch.nn.functional as F

def viewImg(image):
    viewer = ImageViewer(image)
    viewer.show()

def evaluateNumOfAccurates(hypothesis, groundtruth):
    hypothesis = hypothesis.view(-1)
    groundtruth = groundtruth.view(-1)
    accuracy = torch.sum(abs(hypothesis - groundtruth) < 12).item()
    return accuracy

def viewL(tens):
    return tens.view(tens.shape[0], 1, tens.shape[1], tens.shape[2])

def read_image(filename):
    img = io.imread(filename)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], 2)
    return img

def readImgs(imageNamesFile, imageDir):
    imgPaths = open(imageNamesFile, "r").read().split("\n")[:-1] # image paths
    imgPaths = [ (imageDir + sep + x) for x in imgPaths]         # append dir name
    imgs = list(map(read_image, imgPaths))                       # read images
    return imgs

def cvt2Lab(image):
    Lab = color.rgb2lab(image)
    return Lab[:, :, :1], Lab[:, :, 1:]  # L, ab

def readLabs(images):
    lab = list(map(cvt2Lab, images))
    L, ab = ([ x[0] for x in lab ], [ x[1] for x in lab ])
    return L, ab

def cvt2rgb(image):
    return color.lab2rgb(image)

def upsample(image):
    return rescale(image, 4, mode='constant', order=3)

def readImagesToTensors(imageNameList, imageDir):
    # read images
    images = readImgs(imageNameList, imageDir)

    # RGB->Lab conversion
    L, ab = readLabs(images)
    
    # PyTorch tensor
    L  = torch.tensor(L ).float()
    ab = torch.tensor(ab).float()

    L = torch.unsqueeze(L, 1)

    return L, ab

# L: 256x256x1, ab: 64x64x2, out: 256x256x3
def getRGB(L, ab):
    L = L.cpu().detach()
    ab = ab.cpu().detach()

    if(ab.shape[2] == 64):
        ab = F.upsample(ab.view(1, 2, 64, 64), scale_factor=4, mode="bilinear", align_corners=False) # 64->256

    ab = ab.view(256, 256, 2)
    L = L.view(256, 256, 1)

    Lab = np.concatenate( (L.numpy(), ab.numpy()), 2 ).astype(np.float64)
    rgb = cvt2rgb(Lab)
    rgb = rgb * 255
    rgb = rgb.astype(np.int64)
    return rgb
