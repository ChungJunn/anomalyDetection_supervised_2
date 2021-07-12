import sys

def call_model(args, device):
    from ad_model import RNN_enc_RNN_clf, Transformer_enc_RNN_clf, Transformer_enc_DNN_clf, None_enc_DNN_clf, None_enc_RNN_clf
    if args.encoder == 'rnn' and args.classifier == 'rnn':
        model = RNN_enc_RNN_clf(args)
    elif args.encoder == 'transformer' and args.classifier == 'rnn':
        model = Transformer_enc_RNN_clf(args)
    elif args.encoder == "transformer" and args.classifier == "dnn":
        model = Transformer_enc_DNN_clf(args)
    elif args.encoder == "none" and args.classifier == "dnn":
        model = None_enc_DNN_clf(args)
    elif args.encoder == "none" and args.classifier == "rnn":
        model = None_enc_RNN_clf(args)

    else:
        print("encoder and classifier mismatch")
        sys.exit(-1)

    model = model.to(device)

    return model

def get_const(data_name):
    # split to nodes
    if data_name == 'cnsm_exp1_data':
        n_nodes = 5
        n_features = 23
    elif data_name == 'cnsm_exp2_1_data' or data_name == 'cnsm_exp2_2_data':
        n_nodes = 4
        n_features = 23
    elif data_name == "tpi_train_data":
        n_nodes = 5
        n_features = 6
    else:
        print('data_name must be cnsm_exp1_data, cnsm_exp2_1_data, cnsm_exp2_2_data, or tpi_train_data')
        import sys; sys.exit(-1)

    return n_nodes, n_features