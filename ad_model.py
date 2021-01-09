import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class DNN_classifier(nn.Module):
    def __init__(self, dim_input):
        super(DNN_classifier, self).__init__()
        self.fc1 = nn.Linear(dim_input, 600)
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

class pooling_layer:
    def __init__(self, reduce):
        self.reduce = reduce

    def __call__(self, x):
        if self.reduce == 'max':
            layer_out, _ = torch.max(x, dim=0, keepdim=True)
        elif self.reduce == 'mean':
            layer_out = torch.mean(x, dim=0, keepdim=True)
        elif self.reduce == 'last_hidden':
            layer_out = x[-1,:,:]
            layer_out = layer_out.unsqueeze(0)
        else:
            print('reduce must be either \'max\' or \'mean\' or \'last_hidden\'')
            import sys; sys.exit(-1)
        return layer_out
        
#1. reverse
#2. RNN layer
#3. concatenation

class AD_SUP2_MODEL2(nn.Module):
    def __init__(self, dim_lstm_input, dim_lstm_hidden, reduce):
        super(AD_SUP2_MODEL2, self).__init__()
        self.lstm_layer=nn.LSTM(input_size=dim_lstm_input, hidden_size=dim_lstm_hidden)
        self.pooling_layer=pooling_layer(reduce=reduce)
        self.classifier_layer=DNN_classifier(dim_input=dim_lstm_hidden)

    def forward(self, x): 
        # reverse the order
        x = torch.transpose(x, 0, 1)

        # create inverted indices
        idx = [i for i in range(x.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx).to(x.get_device())
        x = x.index_select(0, idx)

        # RNN layer
        x, hidden = self.lstm_layer(x, None)
        #print('lstm h: ', x.shape)
        #print('lstm c: ', hidden[1].shape)

        # pooling layer 
        x = self.pooling_layer(x)
        x = x.squeeze(0) # squeeze node-dimension
        #print('pooling out: ', x.shape)

        # classification layer
        logits = self.classifier_layer(x)
        #print('logits out: ', logits.shape)

        return logits

if __name__ == '__main__':
    mylayer = pooling_layer(reduce='mean')

    myvec = torch.tensor([[1,2,3,4],[1,5,2,3]]).type(torch.float32)

    layer_out = mylayer(myvec)
    
    import pdb; pdb.set_trace()
        

