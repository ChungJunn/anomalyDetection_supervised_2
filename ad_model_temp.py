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