import os
import argparse
from datetime import datetime
import json
import numpy as np
import torch
import random
import torch.nn as nn
import src.utils.interface_logger as logger
import src.data.dataset as dataset
import src.models.model as model_pack
import src.optimizers.optimizer as optimizers
import src.utils.interface_tensorboard as tensorboard
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random_seed = 777
torch.manual_seed(random_seed)
# torch.backends.cudnn.deterministic = True # 연산 속도가 느려질 수 있음
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


def main():
    # Configuration 불러오기
    parser = argparse.ArgumentParser(description='waverdeep - downstream task <speaker-classification>')
    # DISTRIBUTED 사용하기 위해서는 local rank를 argument로 받아야함. 그러면 torch.distributed.launch에서 알아서 해줌
    parser.add_argument("--apex", default=False, type=bool)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--configuration', required=False,
                        default='./config/config_SpeakerClassification_comp_training02.json')
    parser.add_argument("--target_sample", required=False, default=None)
    parser.add_Arguement("--source_sample", required=False, default=None)
    args = parser.parse_args()
    now = datetime.now()
    timestamp = "{}_{}_{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    # read configuration file
    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    target_sample = args.target_sample
    source_sample = args.source_sample

    # create logger
    format_logger = logger.setup_log(save_filename="{}-{}.log".format(config['log_filename'], timestamp))
    # gpu check
    format_logger.info("GPU: {}".format(torch.cuda.is_available()))

    # print configuration 출력
    format_logger.info('configurations: {}'.format(config))

    format_logger.info("load_model ...")
    pretext_model = model_pack.load_model(config, config['pretext_model_name'], config['pretext_checkpoint'])
    downstream_model = model_pack.load_model(config, config['downstream_model_name'])

    # if gpu available: load gpu
    if config['use_cuda']:
        pretext_model = pretext_model.cuda()
        downstream_model = downstream_model.cuda()

    target_waveform = torch.randn((1, 20480))
    source_waveform = torch.randn((1, 20480))

    traget_embeds, source_embeds = inference(config, pretext_model, downstream_model, format_logger, target_waveform, source_waveform)



def inference(config, pretext_model, downstream_model, format_logger, target_waveform, source_waveform):
    pretext_model.eval()
    downstream_model.eval()

    if config['use_cuda']:
        target_waveform = target_waveform.cuda()
        source_waveform = source_waveform.cuda()

    with torch.no_grad():
        loss, accuracy, z, c = pretext_model(target_waveform)
        c = c.detach()
        target_embeds, target_preds = downstream_model(c)

        loss, accuracy, z, c = pretext_model(source_waveform)
        c = c.detach()
        source_embeds, source_preds = downstream_model(c)

    return target_embeds, source_embeds
