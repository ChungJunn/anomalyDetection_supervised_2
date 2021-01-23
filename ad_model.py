import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class RNN_classifier(nn.Module):
    def __init__(self, dim_input, dim_lstm_hidden, dim_fc_hidden, dim_output):
        super(RNN_classifier, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden1 = dim_lstm_hidden
        self.dim_hidden2 = dim_fc_hidden
        self.rnn = nn.LSTM(input_size=dim_input, hidden_size=self.dim_hidden1)
        self.fc1 = nn.Linear(self.dim_hidden1, self.dim_hidden2)
        self.fc2 = nn.Linear(self.dim_hidden2, dim_output)
        self.relu = nn.ReLU()

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)

        output = output.squeeze(0)

        output = self.fc1(output)
        output = self.relu(output)

        output = self.fc2(output)

        return F.log_softmax(output, dim=1), hidden

    def init_hidden(self, Bn):
        hidden = torch.zeros(1, Bn, self.dim_hidden1)
        cell = torch.zeros(1, Bn, self.dim_hidden1)
        return hidden, cell

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
    def __init__(self, dim_input, reduce):
        super(AD_SUP2_MODEL1, self).__init__()

        self.classifier = DNN_classifier(dim_input=dim_input)
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
        annotation=torch.transpose(annotation,0,1)

        encoded = self.encoder(annotation)
        encoded=encoded.squeeze(0)

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

class AD_SUP2_MODEL3(nn.Module):
    def __init__(self, dim_input, nhead, dim_feedforward, reduce, use_feature_mapping, dim_feature_mapping, nlayer):
        super(AD_SUP2_MODEL3, self).__init__()
        if use_feature_mapping:
            d_model = dim_feature_mapping
        else:
            d_model = dim_input

        self.encoder=Transformer_encoder(dim_input, nhead, dim_feedforward, reduce, use_feature_mapping, dim_feature_mapping, nlayer)
        self.classifier=DNN_classifier(dim_input=d_model)

    def forward(self, x):
        x = self.encoder(x)
        logits = self.classifier(x)

        return logits

class RNN_encoder(nn.Module):
    def __init__(self, dim_input, dim_lstm_hidden, reduce, bidirectional, use_feature_mapping, dim_feature_mapping, nlayer, dim_att):
        super(RNN_encoder, self).__init__()
        self.reduce = reduce
        self.use_feature_mapping = use_feature_mapping
        self.dim_feature_mapping = dim_feature_mapping

        if self.reduce == "self-attention":
            if bidirectional == 1:
                dim_att_in = 2 * dim_lstm_hidden
            elif bidirectional == 0:
                dim_att_in = dim_lstm_hidden
            else:
                print("bidirectional must be either 0 or 1")
                import sys; sys.exit(-1)

            self.dim_att = dim_att
            self.att1 = nn.Linear(dim_att_in, self.dim_att)
            self.att2 = nn.Linear(self.dim_att, 1)

        elif self.reduce == 'max' or self.reduce == 'mean':
            self.pooling_layer=pooling_layer(reduce=reduce)
        else:
            print("reduce must be either max, mean, or self-attention")
            import sys; sys.exit(-1)

        # fm layer
        if use_feature_mapping == 1:
            self.fm_layer=nn.Linear(dim_input, dim_feature_mapping)
            dim_lstm_input = dim_feature_mapping
        else:
            dim_lstm_input = dim_input

        if bidirectional == 1:
            self.lstm_layer=nn.LSTM(input_size=dim_lstm_input, hidden_size=dim_lstm_hidden, bidirectional=True, num_layers=nlayer)
        else:
            self.lstm_layer=nn.LSTM(input_size=dim_lstm_input, hidden_size=dim_lstm_hidden, bidirectional=False, num_layers=nlayer)

    def forward(self, x):
        # reverse the order
        x = torch.transpose(x, 0, 1)
        Tx, Bn, D = x.size()

        # create inverted indices
        idx = [i for i in range(x.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx).to(x.get_device())
        x = x.index_select(0, idx)

        # RNN layer
        if self.use_feature_mapping == 1:
            x = x.view(Tx*Bn,D)
            x = self.fm_layer(x)
            x = x.view(Tx,Bn,self.dim_feature_mapping)

        ctx, hidden = self.lstm_layer(x, None)

        # TODO : assume uni-directional rnn
        if self.reduce == "self-attention":
            att1 = torch.tanh(self.att1(ctx))
            att2 = self.att2(att1).view(Tx, Bn)

            alpha = att2 - torch.max(att2)
            alpha = torch.exp(alpha)

            alpha = alpha / (torch.sum(alpha, dim=0, keepdim=True) + 1e-15)
            enc_out = torch.sum(alpha.unsqueeze(2) * ctx, dim=0)
        else:
            out = self.pooling_layer(ctx)
            enc_out = out.squeeze(0) # squeeze node-dimension

        return enc_out

class Transformer_encoder(nn.Module):
    def __init__(self, dim_input, nhead, dim_feedforward, reduce, use_feature_mapping, dim_feature_mapping, nlayer):
        super(Transformer_encoder, self).__init__()
        self.pooling_layer=pooling_layer(reduce=reduce)

        from torch.nn import TransformerEncoderLayer, TransformerEncoder
        self.use_feature_mapping = use_feature_mapping
        self.dim_feature_mapping = dim_feature_mapping

        # use feature mapping
        if self.use_feature_mapping:
            self.fm_layer = nn.Linear(dim_input, dim_feature_mapping)
            d_model = self.dim_feature_mapping
        else:
            d_model = dim_input

        self.t_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.t_layers = TransformerEncoder(encoder_layer=self.t_layer, num_layers=nlayer)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        Tx, Bn, D = x.size()

        if self.use_feature_mapping == 1:
            x = x.contiguous().view(Tx*Bn,D)
            x = self.fm_layer(x)
            x = x.view(Tx,Bn,self.dim_feature_mapping)

        x = self.t_layers(x)

        x = self.pooling_layer(x)
        x = x.squeeze(0)

        return x

class AD_SUP2_MODEL2(nn.Module):
    def __init__(self, dim_input, dim_lstm_hidden, reduce, bidirectional, use_feature_mapping, dim_feature_mapping, nlayer, dim_att):
        super(AD_SUP2_MODEL2, self).__init__()

        if bidirectional==1:
            dim_classifier_input=dim_lstm_hidden*2
        else:
            dim_classifier_input=dim_lstm_hidden

        self.encoder=RNN_encoder(dim_input, dim_lstm_hidden, reduce, bidirectional, use_feature_mapping, dim_feature_mapping, nlayer, dim_att)

        self.classifier=DNN_classifier(dim_input=dim_classifier_input)

    def forward(self, x):
        x = self.encoder(x)
        logits = self.classifier(x)

        return logits

class AD_SUP2_MODEL4(nn.Module):
    #def __init__(self, dim_input, dim_lstm_hidden, dim_fc_hidden, dim_output):
    def __init__(self, dim_input, enc_dim_lstm_hidden, reduce, bidirectional, use_feature_mapping, dim_feature_mapping, nlayer, clf_dim_lstm_hidden, clf_dim_fc_hidden, clf_dim_output):
        super(AD_SUP2_MODEL4, self).__init__()
        if bidirectional==1:
            dim_classifier_input=enc_dim_lstm_hidden*2
        else:
            dim_classifier_input=enc_dim_lstm_hidden

        self.encoder=RNN_encoder(dim_input, enc_dim_lstm_hidden, reduce, bidirectional, use_feature_mapping, dim_feature_mapping, nlayer)

        # classifier
        self.classifier=RNN_classifier(dim_input=dim_classifier_input, dim_lstm_hidden=clf_dim_lstm_hidden, dim_fc_hidden=clf_dim_fc_hidden, dim_output=clf_dim_output)

    def forward(self, x, clf_hidden):
        # through encoder
        x = self.encoder(x)
        x = x.unsqueeze(0)
        logits, clf_hidden = self.classifier(x, clf_hidden)

        return logits, clf_hidden

if __name__ == '__main__':
    mylayer = pooling_layer(reduce='mean')

    myvec = torch.tensor([[1,2,3,4],[1,5,2,3]]).type(torch.float32)

    layer_out = mylayer(myvec)

