import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class DNN_classifier(nn.Module):
    def __init__(self):
        super(DNN_classifier, self).__init__()
        self.fc1 = nn.Linear(22, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 600)
        self.fc4 = nn.Linear(600, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class AD_SUP2_MODEL1(nn.Module):
    def __init__(self, reduce):
        super(AD_SUP2_MODEL1, self).__init__()

        self.classifier = DNN_classifier()
        self.reduce = reduce # either 'mean' or 'max'

    def encoder(self, annotation):
        if self.reduce == 'max':
            enc_out, _ = torch.max(annotation, dim=0, keepdim=True)
        elif self.reduce == 'mean':
            enc_out = torch.mean(annotation, dim=0, keepdim=True)
        else:
            print('reduce must be either \'max\' or \'mean\'')
            import sys; sys.exit(-1)

        return enc_out

    def forward(self, annotation):

        encoded = self.encoder(annotation)
        logits = self.classifier(encoded)

        return logits

