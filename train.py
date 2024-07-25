import torch
import argparse
import shutil
import os, sys
from pathlib import Path
import wandb

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix

if __name__ == '__main__':
    # wandb sweep setting
    sweep_config = {'method': 'random'}
    sweep_config['metric'] = {'name': 'loss', 'goal': 'minimize'}

    parameters_dict = {
      'cascade': {
          'values': [12]
          },
      'chans': {
          'values': [9, 10, 11, 12]
          },
      'sens_chans': {
            'values': [4, 5, 6]
          },
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="varnet-sweep-test")

    # wandb sweep
    wandb.agent(sweep_id, train, count=5)
