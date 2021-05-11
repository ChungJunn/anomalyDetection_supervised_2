import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from libs.layers import PositionalEncoding

class pooling_layer:
    def __init__(self, reduce):
        if reduce not in ['max', 'mean']:
            print('reduce must be either \'max\' or \'mean\'')
            sys.exit(-1)

        self.reduce = reduce

    def __call__(self, x): # x: (Tx, Bn, D)
        if self.reduce == 'max':
            layer_out, _ = torch.max(x, dim=0, keepdim=True)
        elif self.reduce == 'mean':
            layer_out = torch.mean(x, dim=0, keepdim=True)
        return layer_out

class DNN_encoder(nn.Module):
    def __init__(self, dim_input, dim_enc, reduce):
        super(DNN_encoder, self).__init__()
        # encoder
        self.fc1 = nn.Linear(dim_input, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, dim_enc)
        self.relu = nn.ReLU()

        # readout
        self.reduce = reduce

        if self.reduce == "self-attention":
            dim_att_in = dim_enc
            self.dim_att = dim_enc

            self.att1 = nn.Linear(dim_att_in, self.dim_att)
            self.att2 = nn.Linear(self.dim_att, 1)
        elif self.reduce == 'max' or self.reduce == 'mean':
            self.pooling_layer=pooling_layer(reduce=reduce)
        else:
            print("reduce must be either max, mean, or self-attention")
            import sys; sys.exit(-1)

    def forward(self, x):
        # reverse the order
        x = torch.transpose(x, 0, 1).contiguous()
        Tx, Bn, D = x.size()

        # DNN encoder
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        ctx = self.fc3(x)

        if self.reduce == "self-attention":
            att1 = torch.tanh(self.att1(ctx))
            att2 = self.att2(att1).view(Tx, Bn)

            alpha = att2 - torch.max(att2)
            alpha = torch.exp(alpha)

            alpha = alpha / (torch.sum(alpha, dim=0, keepdim=True) + 1e-15)
            enc_out = torch.sum(alpha.unsqueeze(2) * ctx, dim=0) # enc_out: (Bn x D)
        else:
            out = self.pooling_layer(ctx)
            enc_out = out.squeeze(0) # squeeze node-dimension

        return enc_out

class RNN_encoder(nn.Module):
    def __init__(self, dim_input, dim_lstm_hidden, reduce, bidirectional, use_feature_mapping, dim_feature_mapping, nlayer, dim_att):
        super(RNN_encoder, self).__init__()
        self.reduce = reduce
        self.use_feature_mapping = use_feature_mapping
        self.dim_feature_mapping = dim_feature_mapping

        # fm layer
        if use_feature_mapping == 1:
            self.fm_layer = nn.Linear(dim_input, dim_feature_mapping)
            dim_lstm_input = dim_feature_mapping
        else:
            dim_lstm_input = dim_input

        # encoder layer
        if bidirectional == 1:
            self.lstm_layer=nn.LSTM(input_size=dim_lstm_input, hidden_size=dim_lstm_hidden, bidirectional=True, num_layers=nlayer)
        else:
            self.lstm_layer=nn.LSTM(input_size=dim_lstm_input, hidden_size=dim_lstm_hidden, bidirectional=False, num_layers=nlayer)

        # readout
        if self.reduce == 'max' or self.reduce == 'mean':
            self.pooling_layer=pooling_layer(reduce=reduce)

        elif self.reduce == "self-attention":
            if bidirectional == 1:
                dim_att_in = 2 * dim_lstm_hidden
            elif bidirectional == 0:
                dim_att_in = dim_lstm_hidden
            else:
                print("bidirectional must be either 0 or 1")
                import sys; sys.exit(-1)

            self.att1 = nn.Linear(dim_att_in, dim_att)
            self.att2 = nn.Linear(dim_att, 1)
        else:
            print("reduce must be either max, mean, or self-attention")
            import sys; sys.exit(-1)

    def forward(self, x):
        # x: (Bn, V, D), Tx is node-dimension
        # reverse the order
        x = torch.transpose(x, 0, 1).contiguous()
        Tx, Bn, D = x.size()

        # fm
        if self.use_feature_mapping == 1:
            x = x.view(Tx*Bn,D)
            x = self.fm_layer(x)
            x = x.view(Tx,Bn,self.dim_feature_mapping) # x: (V, Bn, D)

        # encoder
        ctx, hidden = self.lstm_layer(x, None) # ctx: (V, Bn, D)

        # readout
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

        return enc_out # (Bn, D)

class RNN_attention_encoder(nn.Module):
    #TODO: use clf_hidden for attention
    def __init__(self, dim_input, dim_lstm_hidden, reduce, bidirectional, use_feature_mapping, dim_feature_mapping, nlayer, dim_att):
        super(RNN_encoder, self).__init__()
        self.reduce = reduce
        self.use_feature_mapping = use_feature_mapping
        self.dim_feature_mapping = dim_feature_mapping

        # fm layer
        if use_feature_mapping == 1:
            self.fm_layer = nn.Linear(dim_input, dim_feature_mapping)
            dim_lstm_input = dim_feature_mapping
        else:
            dim_lstm_input = dim_input

        # encoder layer
        if bidirectional == 1:
            self.lstm_layer=nn.LSTM(input_size=dim_lstm_input, hidden_size=dim_lstm_hidden, bidirectional=True, num_layers=nlayer)
        else:
            self.lstm_layer=nn.LSTM(input_size=dim_lstm_input, hidden_size=dim_lstm_hidden, bidirectional=False, num_layers=nlayer)

        # readout
        if self.reduce == 'max' or self.reduce == 'mean':
            self.pooling_layer=pooling_layer(reduce=reduce)

        elif self.reduce == "self-attention":
            if bidirectional == 1:
                dim_att_in = 2 * dim_lstm_hidden
            elif bidirectional == 0:
                dim_att_in = dim_lstm_hidden
            else:
                print("bidirectional must be either 0 or 1")
                import sys; sys.exit(-1)

            self.att1 = nn.Linear(dim_att_in, dim_att)
            self.att2 = nn.Linear(dim_att, 1)
        else:
            print("reduce must be either max, mean, or self-attention")
            import sys; sys.exit(-1)

    #TODO: use clf_hidden for attention
    def forward(self, x):
        # x: (Bn, V, D), Tx is node-dimension
        # reverse the order
        x = torch.transpose(x, 0, 1).contiguous()
        Tx, Bn, D = x.size()

        # fm
        if self.use_feature_mapping == 1:
            x = x.view(Tx*Bn,D)
            x = self.fm_layer(x)
            x = x.view(Tx,Bn,self.dim_feature_mapping) # x: (V, Bn, D)

        # encoder
        ctx, hidden = self.lstm_layer(x, None) # ctx: (V, Bn, D)

        # readout
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

        return enc_out # (Bn, D)

class Transformer_encoder(nn.Module):
    def __init__(self, dim_input, nhead, dim_feedforward, reduce, use_feature_mapping, dim_feature_mapping, nlayer):
        super(Transformer_encoder, self).__init__()
        self.reduce=reduce
        self.use_feature_mapping = use_feature_mapping
        self.dim_feature_mapping = dim_feature_mapping

        # use feature mapping
        if self.use_feature_mapping:
            self.fm_layer = nn.Linear(dim_input, dim_feature_mapping)
            d_model = self.dim_feature_mapping
        else:
            d_model = dim_input

        # self-attention
        if self.reduce == "self-attention":
            self.dim_att = d_model
            self.dim_att_in = d_model
            self.att1 = nn.Linear(self.dim_att_in, self.dim_att)
            self.att2 = nn.Linear(self.dim_att, 1)
        elif self.reduce == 'max' or self.reduce == 'mean':
            self.pooling_layer=pooling_layer(reduce=reduce)
        else:
            print("reduce must be either max, mean, or self-attention")
            import sys; sys.exit(-1)

        self.positionalEncoding = PositionalEncoding(d_model=d_model)
        self.t_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.t_layers = TransformerEncoder(encoder_layer=self.t_layer, num_layers=nlayer)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        Tx, Bn, D = x.size()

        if self.use_feature_mapping == 1:
            x = x.contiguous().view(Tx*Bn,D)
            x = self.fm_layer(x)
            x = x.view(Tx,Bn,self.dim_feature_mapping)

        x = self.positionalEncoding(x)
        ctx = self.t_layers(x)

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

class DNN_classifier(nn.Module):
    def __init__(self, dim_input, n_fc_layers, dim_fc_hidden, dim_output):
        super(DNN_classifier, self).__init__()
        
        fc_layers = []
        if n_fc_layers < 0:
            print('n_fc_layers must be non-negative')
            sys.exit(-1)
        elif n_fc_layers == 0:
            fc_layers += [nn.Linear(dim_input, dim_output)]
        else:
            fc_layers += [nn.Linear(dim_input, dim_fc_hidden), nn.ReLU()]
            for i in range(n_fc_layers-1):
                fc_layers += [nn.Linear(dim_fc_hidden, dim_fc_hidden), nn.ReLU()]
            fc_layers += [nn.Linear(dim_fc_hidden, dim_output)]
        
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

class RNN_classifier(nn.Module):
    def __init__(self, dim_input, n_lstm_layers, n_fc_layers, dim_lstm_hidden, dim_fc_hidden, dim_output):
        super(RNN_classifier, self).__init__()

        fc_layers = []
        if n_fc_layers == 0:
            fc_layers += [nn.Linear(dim_lstm_hidden, dim_output)]
        else:
            fc_layers += [nn.Linear(dim_lstm_hidden, dim_fc_hidden), nn.ReLU()]
            for i in range(n_fc_layers-1):
               fc_layers +=[nn.Linear(dim_fc_hidden, dim_fc_hidden), nn.ReLU()]
            fc_layers += [nn.Linear(dim_fc_hidden, dim_output)]

        self.rnn = nn.LSTM(input_size=dim_input,
                           hidden_size=dim_lstm_hidden,
                           num_layers=n_lstm_layers)

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x, hidden=None): # x: (Tx x Bn x D)

        x, hidden = self.rnn(x, hidden)

        x = self.fc(x)

        return F.log_softmax(x, dim=2), hidden

class AD_SUP2_MODEL3(nn.Module):
    def __init__(self, args):
        super(AD_SUP2_MODEL3, self).__init__()
        if args.use_feature_mapping:
            d_model = args.dim_feature_mapping
        else:
            d_model = args.dim_input

        self.encoder=Transformer_encoder(args.dim_input, args.nhead, args.dim_feedforward, args.reduce, args.use_feature_mapping, args.dim_feature_mapping, args.nlayer)
        self.classifier=DNN_classifier(dim_input=d_model,
                                       n_fc_layers=args.clf_n_fc_layers,
                                       dim_fc_hidden=args.clf_dim_fc_hidden,
                                       dim_output=args.clf_dim_output)

    def forward(self, x):
        x = self.encoder(x)
        logits = self.classifier(x)

        return logits

class AD_SUP2_MODEL6(nn.Module): # RNN-enc + RNN classifier
    def __init__(self, args):
        super(AD_SUP2_MODEL6, self).__init__()

        if args.bidirectional==1:
            clf_dim_input=args.dim_lstm_hidden*2
        else:
            clf_dim_input=args.dim_lstm_hidden

        # encoder
        self.encoder=RNN_encoder(dim_input = args.dim_input,
                                 dim_lstm_hidden = args.dim_lstm_hidden,
                                 reduce = args.reduce,
                                 bidirectional = args.bidirectional,
                                 use_feature_mapping = args.use_feature_mapping,
                                 dim_feature_mapping = args.dim_feature_mapping,
                                 nlayer = args.nlayer,
                                 dim_att = args.dim_att)

        # classifier
        self.classifier=RNN_classifier(dim_input = clf_dim_input,
                                       n_lstm_layers = args.clf_n_lstm_layers,
                                       n_fc_layers = args.clf_n_fc_layers,
                                       dim_lstm_hidden = args.clf_dim_lstm_hidden,
                                       dim_fc_hidden = args.clf_dim_fc_hidden,
                                       dim_output = args.clf_dim_output)

    def forward(self, x, clf_hidden=None): # (Bn, Tx, V, D) V is number of nodes
        # check batch_size
        if x.shape[0] != 1:
            print('batch_size must be 1 for AD_SUP2_MODEL6')
            sys.exit(-1)

        x = x.squeeze(0) # (Tx, V, D)

        # encoder
        x = self.encoder(x) # (Tx, D) 
        x = x.unsqueeze(1) # (Tx, 1, D)

        logits, clf_hidden = self.classifier(x, clf_hidden) # (Tx, 1, D)
        logits = logits[-1,:,:] # (1, dim_out)

        return logits

class AD_SUP2_MODEL7(nn.Module): # RNN-enc + RNN classifier + attention w/ clf_hidden
    def __init__(self, args, device):
        super(AD_SUP2_MODEL7, self).__init__()
        self.device = device
        # fm layer
        self.use_feature_mapping = args.use_feature_mapping
        self.clf_dim_lstm_hidden = args.clf_dim_lstm_hidden
        if args.use_feature_mapping == 1:
            self.feat_map = nn.Linear(args.dim_input, args.dim_feature_mapping)
            self.d_model = args.dim_feature_mapping
        else:
            self.d_model = args.dim_input

        # encoder
        self.pos_enc = PositionalEncoding(d_model=self.d_model)
        self.trans_enc_layer = TransformerEncoderLayer(d_model=self.d_model,
                                                       nhead=args.nhead,
                                                       dim_feedforward=args.dim_feedforward)
        self.trans_enc = TransformerEncoder(encoder_layer=self.trans_enc_layer,
                                            num_layers=args.nlayer)

        # dec_step
        self.att1 = nn.Linear((self.d_model + args.clf_dim_lstm_hidden), args.dim_att)
        self.att2 = nn.Linear(args.dim_att, 1)
        
        self.rnn_step = nn.LSTMCell(self.d_model, args.clf_dim_lstm_hidden) 

        fc_layers = []
        if args.clf_n_fc_layers == 0:
            fc_layers += [nn.Linear((self.d_model+args.clf_dim_lstm_hidden), args.dim_output)]
        else:
            fc_layers += [nn.Linear((self.d_model+args.clf_dim_lstm_hidden), args.clf_dim_fc_hidden), nn.ReLU()]
            for i in range(args.clf_n_fc_layers-1):
               fc_layers += [nn.Linear(args.clf_dim_fc_hidden, args.clf_dim_fc_hidden), nn.ReLU()]
            fc_layers += [nn.Linear(args.clf_dim_fc_hidden, args.clf_dim_output)]

        self.clf_fc = nn.Sequential(*fc_layers)

    def encoder(self, x):
        x = torch.transpose(x, 0, 1).contiguous()
        Tx, Bn, D = x.size()

        if self.use_feature_mapping == 1:
            x = self.feat_map(x)

        x = self.pos_enc(x) 
        output = self.trans_enc(x)

        return output

    def clf_step(self, h_i, s_htm, s_ctm): # h: (V x d_model), s_tm1: (1 x dim_clf_hidden)
        V, d_model = h_i.size()

        # alignment model
        att_in = torch.cat((h_i, s_htm.expand(V, -1)), dim=1)
        att1 = torch.tanh(self.att1(att_in))
        att2 = self.att2(att1) # V x 1
        att2 = att2 - torch.max(att2)

        alpha = torch.exp(att2)
        alpha = alpha / (torch.sum(alpha, dim=0, keepdim=True) + 1e-15)
        c = torch.sum(alpha * h_i, dim=0, keepdim=True) 

        # clf_rnn
        s_ht, s_ct = self.rnn_step(c, (s_htm, s_ctm)) 

        # clf_dnn
        fc_in = torch.cat((c, s_htm), dim=1)
        fc_out = self.clf_fc(fc_in)
        log_prob = F.log_softmax(fc_out, dim=1)

        return log_prob, s_ht, s_ct

    def forward(self, x_data):
        # check batch_size
        if x_data.shape[0] != 1:
            print('batch_size must be 1 for AD_SUP2_MODEL7')
            sys.exit(-1)

        x_data = x_data.squeeze(0) # (Tx, V, D)

        Tx, V, D = x_data.size()
        h = self.encoder(x_data) # (V, Tx, D)

        s_ht = torch.zeros(1, self.clf_dim_lstm_hidden).type(torch.float32).to(self.device)
        s_ct = s_ht.clone()

        # classifier
        for yi in range(Tx): # use Bn for n_nodes
            h_i = h[:, yi, :]
            log_prob, s_ht, s_ct = self.clf_step(h_i, s_ht, s_ct)

        return log_prob

class AD_SUP2_MODEL8(nn.Module): # Transformer-enc + RNN classifier + attention w/ clf_hidden
    def __init__(self, args, device):
        super(AD_SUP2_MODEL8, self).__init__()
        self.device = device
        # fm layer
        self.use_feature_mapping = args.use_feature_mapping
        self.clf_dim_lstm_hidden = args.clf_dim_lstm_hidden
        if args.use_feature_mapping == 1:
            self.feat_map = nn.Linear(args.dim_input, args.dim_feature_mapping)
            self.d_model = args.dim_feature_mapping
        else:
            self.d_model = args.dim_input

        # encoder
        self.pos_enc = PositionalEncoding(d_model=self.d_model)
        self.trans_enc_layer = TransformerEncoderLayer(d_model=self.d_model,
                                                       nhead=args.nhead,
                                                       dim_feedforward=args.dim_feedforward)
        self.trans_enc = TransformerEncoder(encoder_layer=self.trans_enc_layer,
                                            num_layers=args.nlayer)

        # dec_step
        self.att1 = nn.Linear((self.d_model + args.clf_dim_lstm_hidden), args.dim_att)
        self.att2 = nn.Linear(args.dim_att, 1)
        
        self.rnn_step = nn.LSTMCell(self.d_model, args.clf_dim_lstm_hidden) 

        fc_layers = []
        if args.clf_n_fc_layers == 0:
            fc_layers += [nn.Linear((self.d_model+args.clf_dim_lstm_hidden), args.dim_output)]
        else:
            fc_layers += [nn.Linear((self.d_model+args.clf_dim_lstm_hidden), args.clf_dim_fc_hidden), nn.ReLU()]
            for i in range(args.clf_n_fc_layers-1):
               fc_layers += [nn.Linear(args.clf_dim_fc_hidden, args.clf_dim_fc_hidden), nn.ReLU()]
            fc_layers += [nn.Linear(args.clf_dim_fc_hidden, args.clf_dim_output)]

        self.clf_fc = nn.Sequential(*fc_layers)

    def encoder(self, x):
        x = torch.transpose(x, 0, 1).contiguous()
        Tx, Bn, D = x.size()

        if self.use_feature_mapping == 1:
            x = self.feat_map(x)

        x = self.pos_enc(x) 
        output = self.trans_enc(x)

        return output

    def clf_step(self, h_i, s_htm, s_ctm): # h: (V x d_model), s_tm1: (1 x dim_clf_hidden)
        V, d_model = h_i.size()

        # alignment model
        att_in = torch.cat((h_i, s_htm.expand(V, -1)), dim=1)
        att1 = torch.tanh(self.att1(att_in))
        att2 = self.att2(att1) # V x 1
        att2 = att2 - torch.max(att2)

        alpha = torch.exp(att2)
        alpha = alpha / (torch.sum(alpha, dim=0, keepdim=True) + 1e-15)
        c = torch.sum(alpha * h_i, dim=0, keepdim=True) 

        # clf_rnn
        s_ht, s_ct = self.rnn_step(c, (s_htm, s_ctm)) 

        # clf_dnn
        fc_in = torch.cat((c, s_htm), dim=1)
        fc_out = self.clf_fc(fc_in)
        log_prob = F.log_softmax(fc_out, dim=1)

        return log_prob, s_ht, s_ct

    def forward(self, x_data):
        # check batch_size
        if x_data.shape[0] != 1:
            print('batch_size must be 1 for AD_SUP2_MODEL7')
            sys.exit(-1)

        x_data = x_data.squeeze(0) # (Tx, V, D)

        Tx, V, D = x_data.size()
        h = self.encoder(x_data) # (V, Tx, D)

        s_ht = torch.zeros(1, self.clf_dim_lstm_hidden).type(torch.float32).to(self.device)
        s_ct = s_ht.clone()

        # classifier
        for yi in range(Tx): # use Bn for n_nodes
            h_i = h[:, yi, :]
            log_prob, s_ht, s_ct = self.clf_step(h_i, s_ht, s_ct)

        return log_prob

if __name__ == '__main__':
    mylayer = pooling_layer(reduce='mean')

    myvec = torch.tensor([[1,2,3,4],[1,5,2,3]]).type(torch.float32)

    layer_out = mylayer(myvec)
