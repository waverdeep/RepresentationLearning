import os
import argparse
from datetime import datetime
import json
import numpy as np
import torch
import random
import torch.nn as nn
import sklearn.metrics
import src.utils.interface_logger as logger
import src.data.dataset as dataset
import src.models.model as model_pack
import src.optimizers.optimizer as optimizers
import src.utils.interface_tensorboard as tensorboard
import src.data.dataset_competition as dataset_competition
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
                        default='./config/config_SpeakerClassification_comp_training02_test.json')
    # parser.add_argument("--target_sample", required=False, default='./dataset/speaker_recognition/train/1002/A0046-1002F1412-2__20030-03771376.wav')
    # parser.add_argument("--source_sample", required=False, default='./dataset/speaker_recognition/train/1400/A0050-1400F1111-2__00010-02654579.wav')
    # parser.add_argument("--source_sample", required=False, default='./dataset/speaker_recognition/train/1002/C0603-1002F1412-2__200_0-03273971.wav')
    # parser.add_argument("--source_sample", required=False, default='./dataset/speaker_recognition/train/1002/A0127-1002F1412-2__20010-03793624.wav')
    # parser.add_argument("--source_sample", required=False, default='./dataset/speaker_recognition/train/0762/A0004-0762M1021-10300120-00418404.wav')

    parser.add_argument("--target_sample", required=False, default='./dataset/speaker_recognition/dev/0001/A0002-0001M1113-2__00000-00019659.wav')
    # parser.add_argument("--source_sample", required=False, default='./dataset/speaker_recognition/dev/0001/A0004-0001M1113-2__00000-00019846.wav')
    parser.add_argument("--source_sample", required=False, default='./dataset/speaker_recognition/dev/0031/A0180-0031F1412-10400010-00383574.wav')
    args = parser.parse_args()
    args = parser.parse_args()
    now = datetime.now()
    timestamp = "{}_{}_{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    # read configuration file
    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    target_sample = args.target_sample
    source_sample = args.source_sample

    target_waveform = dataset_competition.load_waveform(target_sample)
    source_waveform = dataset_competition.load_waveform(source_sample)

    # create logger
    format_logger = logger.setup_log(save_filename="{}-{}.log".format(config['log_filename'], timestamp))
    # gpu check
    format_logger.info("GPU: {}".format(torch.cuda.is_available()))

    # print configuration 출력
    format_logger.info('configurations: {}'.format(config))

    format_logger.info("load_model ...")
    pretext_model = model_pack.load_model(config, config['pretext_model_name'], config['pretext_checkpoint'])
    downstream_model = model_pack.load_model(config, config['downstream_model_name'], config['downstream_checkpoint'])

    # if gpu available: load gpu
    if config['use_cuda']:
        pretext_model = pretext_model.cuda()
        downstream_model = downstream_model.cuda()

    # target_waveform = torch.randn((1, 20480))
    # source_waveform = torch.randn((1, 20480))

    target_waveform = target_waveform.unsqueeze(0)
    source_waveform = source_waveform.unsqueeze(0)

    target_embeds, source_embeds = inference(config, pretext_model, downstream_model, format_logger, target_waveform, source_waveform)
    verification_score = calculate_cosine_similarity(target_embeds, source_embeds)
    threshold = 0.8
    if threshold < verification_score:
        status = True
    else:
        status = False
    print("target speaker : ", target_sample.split('/')[4])
    print("source speaker : ", source_sample.split('/')[4])
    print("verification score : ", verification_score)
    print("threshold : ", threshold)
    print("status : ", status)

    target_embeds = target_embeds.cpu().numpy()
    source_embeds = source_embeds.cpu().numpy()
    compute_eer(target_embeds, source_embeds)
    np.save('target_embeds', target_embeds)
    np.save('source_embeds', source_embeds)







def calculate_cosine_similarity(target_embed, source_embed):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(target_embed, source_embed)
    return output.mean()


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


def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


if __name__ == '__main__':
    main()