import os
import argparse
import json
import numpy as np
import torch
import random
import torch.nn as nn
import src.utils.interface_logger as logger
import src.data.dataset as dataset
import src.models.model as model_pack
import src.losses.criterion as losses
import src.optimizers.optimizer as optimizers
import src.utils.interface_tensorboard as tensorboard
import src.utils.interface_train_tool as train_tool
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main():
    # Configuration 불러오기
    parser = argparse.ArgumentParser(description='waverdeep - downstream task <speaker-classification>')
    # DISTRIBUTED 사용하기 위해서는 local rank를 argument로 받아야함. 그러면 torch.distributed.launch에서 알아서 해줌
    parser.add_argument('--configuration', required=False,
                        default='./config/config_spkcls-WAVEBYOL-librispeech100-resnet50aug-batch64.json')
    args = parser.parse_args()
    # read configuration file
    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    now = train_tool.setup_timestamp()

    # create logger
    format_logger = logger.setup_log(save_filename="{}-{}.log".format(config['log_filename'], now))
    # gpu check
    format_logger.info("GPU: {}".format(torch.cuda.is_available()))

    # print configuration 출력
    format_logger.info('configurations: {}'.format(config))
    format_logger.info('load train/test dataset ...')

    # setup train/test dataloader
    train_loader, train_dataset = dataset.get_dataloader(config=config, mode='train')
    test_loader, test_dataset = dataset.get_dataloader(config=config, mode='test')

    format_logger.info("load_model ...")
    pretext_model = model_pack.load_model(config, config['pretext_model_name'], config['pretext_checkpoint'])
    downstream_model = model_pack.load_model(config, config['downstream_model_name'], config['downstream_checkpoint'])

    # setup speaker classfication label
    speaker_dict = train_dataset.speaker_dict
    format_logger.info("speaker_num: {}".format(len(speaker_dict.keys())))

    optimizer = optimizers.get_optimizer(downstream_model.parameters(), config)

    # if gpu available: load gpu
    if config['use_cuda']:
        pretext_model = pretext_model.cuda()
        downstream_model = downstream_model.cuda()

    writer = tensorboard.set_tensorboard_writer(
        "{}-{}".format(config['tensorboard_writer_name'], now)
    )

    # print model information
    format_logger.info(">>> pretext_model_structure <<<")
    model_params = sum(p.numel() for p in pretext_model.parameters() if p.requires_grad)
    format_logger.info("pretext model parameters: {}".format(model_params))
    format_logger.info("{}".format(pretext_model))

    format_logger.info(">>> downstream_model_structure <<<")
    model_params = sum(p.numel() for p in downstream_model.parameters() if p.requires_grad)
    format_logger.info("downstream model parameters: {}".format(model_params))
    format_logger.info("{}".format(downstream_model))

    # start training
    best_accuracy = 0.0
    num_of_epoch = config['epoch']
    for epoch in range(num_of_epoch):
        epoch = epoch + 1
        format_logger.info("start train ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        train_accuracy, train_loss = train(config, writer, epoch, pretext_model, downstream_model, train_loader, optimizer, format_logger,
                                           speaker_dict)
        format_logger.info("start test ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        test_accuracy, test_loss = test(config, writer, epoch, pretext_model, downstream_model,
                                        test_loader, optimizer, format_logger, speaker_dict)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            train_tool.save_checkpoint(config=config, model=downstream_model, optimizer=optimizer,
                                       loss=test_loss, epoch=best_epoch, format_logger=format_logger, mode="best",
                                       date='{}'.format(now))


def train(config, writer, epoch, pretext_model, downstream_model, data_loader, optimizer, format_logger, speaker_dict):
    total_loss = 0.0
    total_accuracy = 0.0
    pretext_model.eval()
    downstream_model.train()
    criterion = losses.set_criterion(config["loss_function"])
    for batch_idx, (waveform, filename, speaker_id) in enumerate(data_loader):
        # 데이터로더의 변경이 필요한가?
        targets = make_target(speaker_id, speaker_dict)
        if config['use_cuda']:
            data = waveform.cuda()
            targets = targets.cuda()
        with torch.no_grad():
            representation = pretext_model.get_representation(data)
        representation = representation.detach()
        B, T, D, C = representation.shape
        # shape 변경 (batch, time, frequency * channel)
        representation = representation.reshape((B, T * C * D))

        predictions = downstream_model(representation)
        loss = criterion(predictions, targets)

        downstream_model.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = torch.zeros(1)
        _, predicted = torch.max(predictions.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        accuracy[0] = correct / total

        writer.add_scalar('Loss/train_step', loss, (epoch - 1) * len(data_loader) + batch_idx)
        writer.add_scalar('Accuracy/train_step', accuracy * 100, (epoch - 1) * len(data_loader) + batch_idx)
        total_loss += len(data) * loss
        total_accuracy += len(data) * accuracy

    total_loss /= len(data_loader.dataset)  # average loss
    total_accuracy /= len(data_loader.dataset)  # average acc

    writer.add_scalar('Loss/train', total_loss, (epoch - 1))
    writer.add_scalar('Accuracy/train', total_accuracy * 100, (epoch - 1))
    return total_accuracy, total_loss


def test(config, writer, epoch, pretext_model, downstream_model, data_loader, optimizer, format_logger, speaker_dict):
    total_loss = 0.0
    total_accuracy = 0.0
    pretext_model.eval()
    downstream_model.eval()
    criterion = losses.set_criterion(config["loss_function"])
    with torch.no_grad():
        for batch_idx, (waveform, filename, speaker_id) in enumerate(data_loader):
            # 데이터로더의 변경이 필요한가?
            targets = make_target(speaker_id, speaker_dict)
            if config['use_cuda']:
                data = waveform.cuda()
                targets = targets.cuda()

            representation = pretext_model.get_representation(data)
            representation = representation.detach()
            B, T, D, C = representation.shape
            # shape 변경 (batch, time, frequency * channel)
            representation = representation.reshape((B, T * C * D))

            predictions = downstream_model(representation)
            loss = criterion(predictions, targets)

            accuracy = torch.zeros(1)
            _, predicted = torch.max(predictions.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            accuracy[0] = correct / total

            writer.add_scalar('Loss/test_step', loss, (epoch - 1) * len(data_loader) + batch_idx)
            writer.add_scalar('Accuracy/test_step', accuracy * 100, (epoch - 1) * len(data_loader) + batch_idx)
            total_loss += len(data) * loss
            total_accuracy += len(data) * accuracy

        total_loss /= len(data_loader.dataset)  # average loss
        total_accuracy /= len(data_loader.dataset)  # average acc

        writer.add_scalar('Loss/test', total_loss, (epoch - 1))
        writer.add_scalar('Accuracy/test', total_accuracy * 100, (epoch - 1))
    return total_accuracy, total_loss


def make_target(speaker_id, speaker_dict):
    targets = torch.zeros(len(speaker_id)).long()
    for idx in range(len(speaker_id)):
        targets[idx] = speaker_dict[speaker_id[idx]]
    return targets


if __name__ == '__main__':
    main()