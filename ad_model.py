import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class AD_SUP2_MODEL1(nn.Module):
    def __init__(self, reduce):
        super(AD_SUP2_MODEL1, self).__init__()

        self.fc1 = nn.Linear(22, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 600)
        self.fc4 = nn.Linear(600, 2)
        self.relu = nn.ReLU()

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

    def classifier(self, enc_out):
        a1 = self.fc1(enc_out)
        h1 = self.relu(a1)

        a2 = self.fc2(h1)
        h2 = self.relu(a2)

        a3 = self.fc3(h2)
        h3 = self.relu(a3)

        out = self.fc4(h3)

        return F.log_softmax(out, dim=1)

    def forward(self, annotation):

        encoded = self.encoder(annotation)
        logits = self.classifier(encoded)

        return logits

