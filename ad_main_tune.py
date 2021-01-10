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
    parser.add_argument('--reduce', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--patience', type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--encoder', type=str)
    # Simple model params
    parser.add_argument('--dim_input', type=int)
    # RNN params
    parser.add_argument('--dim_lstm_input', type=int)
    parser.add_argument('--dim_lstm_hidden', type=int)
    parser.add_argument('--bidirectional', type=int)
    # Transformer params 
    parser.add_argument('--d_model', type=int)
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--dim_feedforward', type=int)

    args = parser.parse_args()

    args.out_file='temp.tune.pth'

    config = {
        'optimizer':tune.choice(['Adam','SGD','RMSprop']),
        'lr':tune.qloguniform(1e-4,0.1,1e-4),
        'batch_size':tune.choice([32,64,128]),
        'reduce':tune.choice(['mean','max'])
    }

    asha_scheduler=ASHAScheduler(
        metric='val_f1',
        mode='max',
        grace_period=5,
        max_t=args.max_epoch
        )

    reporter=CLIReporter(
        metric_columns=["train_loss","val_f1"]
        )

    experiment=tune.run(
        partial(main,args=args),
        name='raytune_none',
        config=config,
        local_dir='none_result',
        resources_per_trial={'cpu':2,'gpu':0.25},
        num_samples=200,
        scheduler=asha_scheduler,
        fail_fast=True,
        log_to_file=True,
        progress_reporter=reporter)
