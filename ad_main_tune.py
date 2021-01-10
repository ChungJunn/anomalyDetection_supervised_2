from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

from ad_main import train_main

def main(config, checkpoint_dir=None, args=None):
    args.lr=config['lr']
    args.dim1=config['dim1']
    args.dim2=config['dim2']
    args.wd=config[

    neptune=None

    train_main(args, neptune)
