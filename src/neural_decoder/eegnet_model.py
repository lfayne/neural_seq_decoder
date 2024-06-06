import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchaudio.transforms as T
import numpy as np

class EEGNet(nn.Module):
    def __init__(self, neural_dim=256, n_classes=40, sfreq=50, dropout=0.5, strideLen=4, kernelLen=32, nDays=24):
        super(EEGNet, self).__init__()

        self.neural_dim = neural_dim
        self.sfreq = sfreq
        self.nDays = nDays
        self.n_classes = n_classes
        
        self.strideLen=strideLen
        self.kernelLen=kernelLen
        
        self.inputLayerNonlinearity = torch.nn.Softsign()
            
        self.day_weights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.day_biases = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.day_weights.data[x, :, :] = torch.eye(neural_dim)

        # Layer 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(16)

        # Layer 2: Depthwise Convolution + Separable Convolution
        self.conv2 = nn.Conv2d(16, 32, (neural_dim, 1), groups=16)  # Depthwise
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (1, 16), padding='same')  # Separable
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.avgpool1 = nn.AvgPool2d((1, 4))  # Adjust pooling size if needed
        self.dropout1 = nn.Dropout(dropout)

        # Layer 3: Separable Convolution
        self.conv4 = nn.Conv2d(32, 64, (1, 16), padding='same')
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.avgpool2 = nn.AvgPool2d((1, 8))  # Adjust pooling size if needed
        self.dropout2 = nn.Dropout(dropout)

        # Classifier
        self.flatten = nn.Flatten()
        #self.classifier = nn.Linear(64 * (int(np.ceil(1200/32))-15), n_classes + 1) # Adjust for different input size
        #self.classifier = nn.Linear(64 * 2, n_classes + 1) # Adjust for different input size
        self.classifier = nn.Linear(1024 * 2, n_classes + 1) # Adjust for different input size
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, neuralInput, day_idx):

        # apply day layer
        day_weights = torch.index_select(self.day_weights, 0, day_idx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, day_weights
        ) + torch.index_select(self.day_biases, 0, day_idx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        
        transformedNeural = transformedNeural.unsqueeze(1) 

        neuralInput = F.elu(self.conv1(transformedNeural))
        neuralInput = self.batchnorm1(neuralInput)
        neuralInput = F.elu(self.conv2(neuralInput))
        neuralInput = self.batchnorm2(neuralInput)
        neuralInput = F.elu(self.conv3(neuralInput))
        neuralInput = self.batchnorm3(neuralInput)
        neuralInput = self.avgpool1(neuralInput)
        neuralInput = self.dropout1(neuralInput)
        neuralInput = F.elu(self.conv4(neuralInput))
        neuralInput = self.batchnorm4(neuralInput)
        neuralInput = self.avgpool2(neuralInput)
        neuralInput = self.dropout2(neuralInput)
        neuralInput = self.flatten(neuralInput)
        
        # Dynamically calculate classifier input size
        classifier_input_size = neuralInput.shape[1]  # Get the size of the flattened dimension

        # Recreate the classifier with the correct input size
        self.classifier = nn.Linear(classifier_input_size, self.n_classes + 1)

        neuralInput = self.classifier(neuralInput)
        neuralInput = self.softmax(neuralInput)
        
        return neuralInput