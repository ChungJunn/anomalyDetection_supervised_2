from ad_model import RNN_enc_RNN_clf, Transformer_enc_RNN_clf, Transformer_enc_DNN_clf
import sys

def call_model(args, device):
    if args.encoder == 'rnn' and args.classifier == 'rnn':
        model = RNN_enc_RNN_clf(args)
    elif args.encoder == 'transformer' and args.classifier == 'rnn':
        model = Transformer_enc_RNN_clf(args)
    elif args.encoder == "transformer" and args.classifier == "dnn":
        model = Transformer_enc_DNN_clf(args)
    else:
        print("encoder and classifier mismatch")
        sys.exit(-1)

    model = model.to(device)

    return model