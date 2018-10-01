#####################################
# METU - CENG483 HW3                #
# Author:                           #
#  Necip Fazil Yildiran             #
#####################################

import matplotlib.pyplot as plt
import numpy as np

# Draw the loss history plot and save it to the file 'loss_history.png'
def plotLossHist(trainLossHist, validLossHist, params):
    numOfEpochs = len(trainLossHist)
    
    # max(lossHistory) could be used to capture all the loss history
    # .. however, for making comparison easier and outputting a more precise
    # .. plot, use a constant
    yMax = max([max(trainLossHist), max(validLossHist)]) + 30
    yMin = min([min(trainLossHist), min(validLossHist)]) - 10
    horizontalSpace = numOfEpochs + numOfEpochs * 0.12
    plt.axis([0, horizontalSpace, yMin, yMax])

    xValues = np.arange(1, numOfEpochs + 1, 1)
    plt.plot(xValues, trainLossHist, label='Loss History on Training Data', c='b', linewidth = 1.0)
    plt.plot(xValues, validLossHist, label='Loss History on Validation Data', c='r', linewidth = 1.0)

    # annotate last values
    plt.annotate('{:.2f}'.format(trainLossHist[-1]), xy=(numOfEpochs, trainLossHist[-1]), color='blue')
    plt.annotate('{:.2f}'.format(validLossHist[-1]), xy=(numOfEpochs, validLossHist[-1]), color='red')

    (learningRate, reguStr, batchsize) = params
    hyperparameters_str = ' Number of Epochs = ' + str(numOfEpochs)
    hyperparameters_str += '\n Learning Rate = ' + learningRate
    hyperparameters_str += '\n Weight Decay = ' + reguStr
    hyperparameters_str += '\n Batch Size = ' + batchsize

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
    plt.text(horizontalSpace / 1.55, yMax * 0.93, hyperparameters_str, horizontalalignment='left', verticalalignment='top', bbox = props)

    plt.legend()

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title("Loss History on Training and Validation Data")
    
    plt.savefig('loss_history.png')
    plt.close('all')
