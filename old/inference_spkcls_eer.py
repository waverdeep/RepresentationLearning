import os
import argparse
from datetime import datetime
import json
import numpy as np
import torch
import random
from torch.utils import data
import torch.nn as nn
import sklearn.metrics
import src.utils.interface_logger as logger
import src.data.dataset as dataset
import src.models.model as model_pack
import src.optimizers.optimizer as optimizers
import src.utils.interface_tensorboard as tensorboard
import src.data.dataset_competition as dataset_competition
import src.utils.interface_file_io as file_io
import src.data.dataset_competition as competition
from tqdm import tqdm
import time
from scipy.spatial import distance
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
    parser.add_argument("--test_trials_path", required=False, default='./dataset/test_trails_edit.txt')
    parser.add_argument("--dataset_path", required=False, default='./dataset/')
    # parser.add_argument("--target_sample", required=False,
    #                     default="./dataset/speaker_recognition/dev/0081/A0185-0081F1112-10510110-00107255.wav")
    # parser.add_argument("--source_sample", required=False,
    #                     default=["./dataset/speaker_recognition/dev/0081/A0185-0081F1112-10510110-00107314.wav",
    #                              "./dataset/speaker_recognition/dev/0081/A0186-0081F1112-10510110-00107380.wav",
    #                              "./dataset/speaker_recognition/dev/0081/A0186-0081F1112-10510110-00107434.wav",
    #                              "./dataset/speaker_recognition/dev/0081/A0186-0081F1112-10510110-00107490.wav",
    #                              "./dataset/speaker_recognition/dev/0115/A0130-0115F1110-10200010-00177420.wav",
    #                              "./dataset/speaker_recognition/dev/0158/A0052-0158M1112-10200000-03523755.wav",
    #                              "./dataset/speaker_recognition/dev/0162/A0145-0162F1311-2__00010-00086225.wav",
    #                              "./dataset/speaker_recognition/dev/0180/C0236-0180F1222-102000_0-01046142.wav"])

    args = parser.parse_args()
    now = datetime.now()
    timestamp = "{}_{}_{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    # read configuration file
    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    dataset = file_io.read_txt2list(args.test_trials_path)
    print(len(dataset))
    filelist = file_io.get_all_file_path(args.dataset_path, 'wav')


    target_samples = []
    source_samples = []
    positives = []
    for index, line in enumerate(dataset):
        if index % 1000 == 0:
            print("reading ... {}".format(index))
        data_line = line.split(' ')
        target_samples.append("{}{}".format(args.dataset_path, data_line[0].replace("test", "test_wav")))
        source_samples.append("{}{}".format(args.dataset_path, data_line[0].replace("test", "test_wav")))
        if data_line[2] == 'target':
            positives.append(1)
        elif data_line[2] == 'nontarget':
            positives.append(0)
    print(target_samples[:5])
    print(source_samples[:5])
    print(positives[:5])

    # target_sample = args.target_sample
    # source_sample = args.source_sample

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

    predictions = []

    loop_cout = len(source_samples)
    print("sample count: ", loop_cout)

    target_dataset = competition.CompetitionInferenceWaveform(file_list=target_samples)
    source_dataset = competition.CompetitionInferenceWaveform(file_list=source_samples)
    target_dataloader = data.DataLoader(
        dataset=target_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )
    source_dataloader = data.DataLoader(
        dataset=source_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )
    target_embeds, index_list = inference_batch(config, pretext_model, downstream_model, format_logger,
                                                target_dataloader)
    source_embeds, index_list = inference_batch(config, pretext_model, downstream_model, format_logger,
                                                source_dataloader)

    targets = torch.cat(target_embeds, dim=0)
    sources = torch.cat(source_embeds, dim=0)
    data_len = len(targets)
    for index in tqdm(range(loop_cout), desc='calc eer'):
        verification_score = calculate_cosine_similarity(targets[index], sources[index])
        predictions.append(float(verification_score.cpu().numpy()))
    eer = compute_eer(positives, predictions)
    print("EER: ", eer)


    # for index in tqdm(range(loop_cout), desc='inference'):
    #     target_waveform = dataset_competition.load_waveform(target_samples[index])
    #     target_waveform = target_waveform.unsqueeze(0)
    #     target_embeds = inference(config, pretext_model, downstream_model, format_logger, target_waveform)
    #
    #     source_waveform = dataset_competition.load_waveform(source_samples[index])
    #     source_waveform = source_waveform.unsqueeze(0)
    #     source_embeds = inference(config, pretext_model, downstream_model, format_logger, source_waveform)
    #     verification_score = calculate_cosine_similarity(target_embeds, source_embeds)
    #     predictions.append(float(verification_score.cpu().numpy()))

    # eer = compute_eer(positives, predictions)
    # print("EER: ", eer)
    #
    # with open("my_result.txt", 'w') as result:
    #     result.write(eer)
    # output_embeddings = np.ndarray(predictions)
    # np.save("result_embeddings", output_embeddings)



def calculate_cosine_similarity(target_embed, source_embed):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    output = cos(target_embed, source_embed)
    return output.mean()


def inference(config, pretext_model, downstream_model, format_logger, waveform):
    pretext_model.eval()
    downstream_model.eval()
    index_list = []

    with torch.no_grad():
        if config['use_cuda']:
            data = waveform.cuda()

        loss, accuracy, z, c = pretext_model(data)
        c = c.detach()
        embeds, preds = downstream_model(c)

    return embeds


def inference_batch(config, pretext_model, downstream_model, format_logger, dataloader):
    pretext_model.eval()
    downstream_model.eval()
    index_list = []
    batch_list = []
    with torch.no_grad():
        for batch_idx, (waveform, filename, speaker_id) in enumerate(dataloader):
            if config['use_cuda']:
                data = waveform.cuda()

            loss, accuracy, z, c = pretext_model(data)
            c = c.detach()
            embeds, preds = downstream_model(c)
            index_list.append(filename)
            batch_list.append(embeds)


    return batch_list, index_list


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