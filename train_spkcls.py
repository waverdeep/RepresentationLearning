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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
                        default='./config/config_SpeakerClassification_training05.json')
    args = parser.parse_args()
    now = datetime.now()
    timestamp = "{}_{}_{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    # read configuration file
    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    # create logger
    format_logger = logger.setup_log(save_filename="{}-{}.log".format(config['log_filename'], timestamp))
    # gpu check
    format_logger.info("GPU: {}".format(torch.cuda.is_available()))

    # print configuration 출력
    format_logger.info('configurations: {}'.format(config))
    format_logger.info('load train/test dataset ...')

    # setup train/test dataloader
    train_loader, train_dataset = dataset.get_dataloader(config=config, mode='train')
    test_loader, test_dataset = dataset.get_dataloader(config=config, mode='test')

    # setup speaker classfication label
    speaker_dict = train_dataset.speaker_dict
    format_logger.info("speaker_num: {}".format(len(speaker_dict.keys())))

    format_logger.info("load_model ...")
    pretext_model = model_pack.load_model(config, config['pretext_model_name'], config['pretext_checkpoint'])
    downstream_model = model_pack.load_model(config, config['downstream_model_name'])

    optimizer = optimizers.get_optimizer(downstream_model.parameters(), config)

    # if gpu available: load gpu
    if config['use_cuda']:
        pretext_model = pretext_model.cuda()
        downstream_model = downstream_model.cuda()

    writer = tensorboard.set_tensorboard_writer(
        "{}-{}_{}_{}_{}_{}_{}".format(config['tensorboard_writer_name'],
                                      now.year, now.month, now.day, now.hour,
                                      now.minute, now.second)
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
        train(config, writer, epoch, pretext_model, downstream_model, train_loader, optimizer, format_logger,
              speaker_dict)
        format_logger.info("start test ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        test_accuracy, test_loss = test(config, writer, epoch, pretext_model, downstream_model,
                                        test_loader, optimizer, format_logger, speaker_dict)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            save_checkpoint(config=config, model=downstream_model, optimizer=optimizer,
                            loss=test_loss, epoch=best_epoch, format_logger=format_logger, mode="best",
                            date='{}_{}_{}_{}_{}_{}'.format(now.year, now.month, now.day, now.hour,
                                                            now.minute, now.second))


def train(config, writer, epoch, pretext_model, downstream_model, train_loader, optimizer, format_logger, speaker_dict):
    pretext_model.eval()
    downstream_model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_accuracy = 0.0
    format_logger.info("[ {}/{} epoch train result: [ average acc: {}/ average loss: {} ]".format(
        epoch, config['epoch'], total_accuracy, total_loss
    ))
    for batch_idx, (waveform, filename, speaker_id) in enumerate(train_loader):
        targets = make_target(speaker_id, speaker_dict)
        if config['use_cuda']:
            data = waveform.cuda()
            targets = targets.cuda()
        with torch.no_grad():
            loss, accuracy, z, c = pretext_model(data)
        # targets = torch.nn.functional.one_hot(targets, num_classes=251)
        c = c.detach()
        preds = downstream_model(c)
        loss = criterion(preds, targets)

        downstream_model.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = torch.zeros(1)
        _, predicted = torch.max(preds.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        accuracy[0] = correct/total

        writer.add_scalar('Loss/train_step', loss, (epoch - 1) * len(train_loader) + batch_idx)
        writer.add_scalar('Accuracy/train_step', accuracy * 100, (epoch - 1) * len(train_loader) + batch_idx)
        total_loss += len(data) * loss
        total_accuracy += len(data) * accuracy

    total_loss /= len(train_loader.dataset)  # average loss
    total_accuracy /= len(train_loader.dataset)  # average acc

    writer.add_scalar('Loss/train', total_loss, (epoch - 1))
    writer.add_scalar('Accuracy/train', total_accuracy * 100, (epoch - 1))


def test(config, writer, epoch, pretext_model, downstream_model, test_loader, optimizer, format_logger, speaker_dict):
    pretext_model.eval()
    downstream_model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_accuracy = 0.0

    format_logger.info("[ {}/{} epoch test result: [ average acc: {}/ average loss: {} ]".format(
        epoch, config['epoch'], total_accuracy, total_loss
    ))

    with torch.no_grad():
        for batch_idx, (waveform, filename, speaker_id) in enumerate(test_loader):
            targets = make_target(speaker_id, speaker_dict)
            if config['use_cuda']:
                data = waveform.cuda()
                targets = targets.cuda()

            loss, accuracy, z, c = pretext_model(data)
            # targets = torch.nn.functional.one_hot(targets, num_classes=251)
            c = c.detach()
            preds = downstream_model(c)
            loss = criterion(preds, targets)

            accuracy = torch.zeros(1)
            _, predicted = torch.max(preds.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            accuracy[0] = correct / total

            writer.add_scalar('Loss/test_step', loss, (epoch - 1) * len(test_loader) + batch_idx)
            writer.add_scalar('Accuracy/test_step', accuracy * 100, (epoch - 1) * len(test_loader) + batch_idx)
            total_loss += len(data) * loss
            total_accuracy += len(data) * accuracy

        total_loss /= len(test_loader.dataset)  # average loss
        total_accuracy /= len(test_loader.dataset)  # average acc

        writer.add_scalar('Loss/test', total_loss, (epoch - 1))
        writer.add_scalar('Accuracy/test', total_accuracy * 100, (epoch - 1))
    return total_accuracy, total_loss


def make_target(speaker_id, speaker_dict):
    targets = torch.zeros(len(speaker_id)).long()
    for idx in range(len(speaker_id)):
        targets[idx] = speaker_dict[speaker_id[idx]]
    return targets


def save_checkpoint(config, model, optimizer, loss, epoch, format_logger, mode="best", date=""):
    if mode == "best":
        file_path = os.path.join(config['checkpoint_save_directory_path'],
                                 config['checkpoint_file_name'] + "-model-best-{}.pt".format(date))

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "loss": loss}, file_path)
    format_logger.info("saved best checkpoint ... ")


if __name__ == '__main__':
    main()
