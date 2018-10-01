#####################################
# METU - CENG483 HW3                #
# Author:                           #
#  Necip Fazil Yildiran             #
#####################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from os import sep
import utils
import plotutils

def evalLossAcc(net, bulk_valid_256_L, bulk_valid_64_ab, bulk_valid_256_rgb):
    # mean squared error loss
    criterion = nn.MSELoss()

    accuracyAccumulator = 0
    lossAccumulator = 0

    for inp, gt_64, gt_256 in zip(bulk_valid_256_L, bulk_valid_64_ab, bulk_valid_256_rgb):
        inp = torch.Tensor(inp).cuda().unsqueeze(0).view(1, 1, 256, 256)
        gt_64 = torch.Tensor(gt_64).cuda().view(2, 64, 64)
        gt_256 = torch.Tensor(gt_256).view(3, 256, 256)

        hypo_64_ab = net(inp)

        # find hypo rgb by making use of gt L channel and hypo ab channels
        hypo_rgb = torch.Tensor(utils.getRGB(inp, hypo_64_ab)).cpu()

        # accuracy
        accuracyAccumulator = accuracyAccumulator + utils.evaluateNumOfAccurates(hypo_rgb, gt_256)

        # loss
        lossAccumulator = lossAccumulator + criterion(hypo_64_ab[0], gt_64).item()

        # free memory on GPU
        del inp, gt_64, gt_256
    
    # compute final accuracy and loss
    accuracy = accuracyAccumulator / float( len(bulk_valid_256_rgb) * 256 * 256 * 3)
    loss = lossAccumulator / float( len(bulk_valid_256_rgb) )

    return loss, accuracy

def train(net, data, numOfEpochs, learningRate, regularizationStrength, batchsize):
    # resolve data
    train_256_L, train_256_ab, train_64_L, train_64_ab, valid_256_L, valid_256_ab, valid_64_L, valid_64_ab, valid_256_rgb = data

    # Loss func: MeanSquaredError
    criterion = nn.MSELoss()

    # Optimizer: RMSprop
    optimizer = optim.RMSprop(net.parameters(), lr=learningRate, weight_decay=regularizationStrength)

    # history lists
    trainLossHistory = []
    validLossHistory = []
    validAccuracyHistory = []

    print("Training is starting. It will take " + str(numOfEpochs) + " epochs and will dump information at the end of each epoch.")
    for epoch in range(0, numOfEpochs):
        # accumulate the loss of each mini-batch inside
        trainLossAcc = 0

        # use data loader to create mini-batches
        train_L_loader = DataLoader(train_256_L, batch_size=batchsize)
        train_ab_loader = DataLoader(train_64_ab, batch_size=batchsize)

        for train_input_L_ram, train_gt_ab_ram in zip(train_L_loader, train_ab_loader):
            # move mini batch to GPU for better performance
            train_input_L = train_input_L_ram.cuda().view(-1, 1, 256, 256)
            train_gt_ab = train_gt_ab_ram.cuda()

            # obtain hypothesis using the current state of network
            train_hypothesis = net(train_input_L)
            train_hypothesis = train_hypothesis.view(train_gt_ab.size()) 

            # compute loss
            trainLossTensor = criterion(train_hypothesis, train_gt_ab)

            # accumulate loss
            trainLossAcc = trainLossAcc + trainLossTensor.item()

            # reset optimizer's grad space to all zeros
            optimizer.zero_grad()

            # for gradient computation, apply backward()
            trainLossTensor.backward()

            # perform optimization on network model
            optimizer.step()

            # free memory on GPU
            del train_input_L, train_gt_ab

        # compute average loss for the current epoch
        trainLoss = float(trainLossAcc) / (train_256_L.shape[0] / float(batchsize))

        #######################################
        # Loss computation on validation data #
        #######################################
        validLoss, validAccuracy = evalLossAcc(net, valid_256_L, valid_64_ab, valid_256_rgb)

        # append the loss values to the loss history
        trainLossHistory.append(trainLoss)
        validLossHistory.append(validLoss)
        validAccuracyHistory.append(validAccuracy)

        # Write the epoch&loss
        print("Epoch [%3d]: \tTrainLoss: %7.2f, ValidLoss: %7.2f, ValidAcc: %3.2f"
                %(epoch, trainLoss, validLoss, validAccuracy))
            
        # to avoid last optimization that is not recorded
        if(epoch + 1 == numOfEpochs):
            break

    # output last training statistic
    print("Training is done with %d epochs. \n\tTrainLoss: %7.2f, ValidLoss: %7.2f, ValidAcc: %3.2f" %(numOfEpochs, trainLossHistory[-1], validLossHistory[-1], validAccuracyHistory[-1]))

    return trainLossHistory, validLossHistory, validAccuracyHistory

if __name__=='__main__':
    print("Loading images..")

    # rgb for accuracy computation
    valid_256_rgb = utils.readImgs("valid.txt", "color_256")

    # all data on Lab space
    train_256_L, train_256_ab = utils.readImagesToTensors("train.txt",  "color_256")
    train_64_L,  train_64_ab  =  utils.readImagesToTensors("train.txt", "color_64")

    valid_256_L, valid_256_ab = utils.readImagesToTensors("valid.txt",  "color_256")
    valid_64_L,  valid_64_ab  = utils.readImagesToTensors("valid.txt",  "color_64")

    test_256_L,  _            = utils.readImagesToTensors("test.txt",   "gray")
    print("Images loaded.")

    # pack data - to be passed to other functions
    data = (train_256_L, train_256_ab, train_64_L, train_64_ab, valid_256_L, valid_256_ab, valid_64_L, valid_64_ab, valid_256_rgb)

    # create network
    net = Net().cuda()

    # config of training
    numOfEpochs = 5
    learningRate = 1e-3
    regularizationStr = 1e-3

    # train!
    trainLossHist, validLossHist, validAccHist = train(net, data, numOfEpochs, learningRate, regularizationStr, batchsize=50)

    # plotting function could be used exactly as follows.
    # it outputs the plot as an image file, loss_history.png, in the current directory
    # plotutils.plotLossHist(trainLossHist, validLossHist, ("1e-3", "1e-3", "50")) # lr, wd, bs

    # test
    print("Drawing estimations on test data..")
    test_hypo_ab = net(test_256_L.cuda().view(-1, 1, 256, 256))
    
    # construct rgb images
    rgbs = list(map(utils.getRGB, test_256_L, test_hypo_ab))

    # save to estimations.npy
    np.save("estimations.npy", rgbs)
    print("Estimations are saved to estimations.npy.")
