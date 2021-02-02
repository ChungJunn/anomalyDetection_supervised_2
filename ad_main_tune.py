from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

from ad_main import train_main

def main(config, checkpoint_dir=None, args=None):
    args.optimizer=config['optimizer']
    args.lr=config['lr']
    args.batch_size=config['batch_size']
    args.reduce=config['reduce']
    args.nlayer=config['nlayer']

    if args.encoder=='rnn' or args.encoder=='bidirectionalrnn':
        args.dim_lstm_hidden=config['dim_lstm_hidden']
        args.dim_att=config['dim_lstm_hidden']
        args.dim_feature_mapping=config['dim_feature_mapping']
    elif args.encoder=='transformer':
        args.nhead=config['nhead']
        args.dim_feedforward=config['dim_feedforward']
        args.dim_feature_mapping=config['dim_feature_mapping']

    neptune=None

    train_main(args, neptune)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--csv1', type=str)
    parser.add_argument('--csv2', type=str)
    parser.add_argument('--csv3', type=str)
    parser.add_argument('--csv4', type=str)
    parser.add_argument('--csv5', type=str)
    parser.add_argument('--csv_label', type=str)
    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--patience', type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--tune', type=int)
    parser.add_argument('--test_log_interval', type=int)
    # simple model params
    parser.add_argument('--dim_input', type=int)
    parser.add_argument('--use_feature_mapping', type=int)
    # RNN params
    parser.add_argument('--bidirectional', type=int)

    args = parser.parse_args()

    name = args.dataset + '.' + args.encoder

    if args.encoder=='none':
        config = {
            'optimizer':tune.choice(['Adam','SGD','RMSprop']),
            'lr':tune.qloguniform(1e-4,0.1,1e-4),
            'batch_size':tune.choice([32,64,128]),
            'reduce':tune.choice(['mean','max']),
            'nlayer':tune.choice([1,2,3])
        }
    elif args.encoder=='rnn' or args.encoder=='bidirectionalrnn':
        config = {
            'optimizer':tune.choice(['Adam','SGD','RMSprop']),
            'lr':tune.qloguniform(1e-4,0.1,1e-4),
            'batch_size':tune.choice([32,64,128]),
            'reduce':tune.choice(['self-attention']),
            'nlayer':tune.choice([1,2,3]),
            'dim_lstm_hidden':tune.choice([8, 16, 32]),
            'dim_feature_mapping':tune.choice([12,24,36])
        }
    elif args.encoder=='transformer':
        config = {
            'optimizer':tune.choice(['Adam','SGD','RMSprop']),
            'lr':tune.qloguniform(1e-4,0.1,1e-4),
            'batch_size':tune.choice([32,64,128]),
            'reduce':tune.choice(['self-attention']),
            'nlayer':tune.choice([1,2,3]),
            'nhead':tune.choice([2,3,6]),
            'dim_feedforward':tune.choice([32,64,128]),
            'dim_feature_mapping':tune.choice([12,24,36])
        }

    asha_scheduler=ASHAScheduler(
        metric='val_f1',
        mode='max',
        grace_period=5,
        max_t=args.max_epoch
        )

    reporter=CLIReporter(
        metric_columns=["train_loss","val_f1","test_f1"]
        )

    experiment=tune.run(
        partial(main,args=args),
        name=name,
        config=config,
        local_dir='.ray_result',
        resources_per_trial={'cpu':2,'gpu':0.25},
        num_samples=args.n_samples,
        scheduler=asha_scheduler,
        fail_fast=True,
        log_to_file=True,
        progress_reporter=reporter)
