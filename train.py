import argparse
import json
import random
import numpy as np
import os
import torch.cuda
import src.optimizers.optimizer as optimizers
import src.data.dataset as dataset
import src.models.model as model_pack
import src.utils.logger as logger
import src.utils.setup_tensorboard as tensorboard
from apex.parallel import DistributedDataParallel as DDP
from datetime import datetime
import src.utils.plots as plots


def setup_seed(random_seed):
    torch.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True # 연산 속도가 느려질 수 있음
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def setup_argparse(description, use_apex=False):
    parser = argparse.ArgumentParser(description='waverdeep - representation learning')
    parser.add_argument('--configuration',
                        required=False, default='./config/config_CPC_baseline_training01-batch24.json')
    if use_apex:
        parser.add_argument("--apex", default=False, type=bool)
        parser.add_argument("--local_rank", default=0, type=int)

    return parser.parse_args()


def setup_timestamp():
    now = datetime.now()
    return "{}_{}_{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)


def setup_distributed_learning(config, format_logger):
    pass





