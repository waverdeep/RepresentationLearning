import src.utils.interface_train_tool as train_tool
import src.utils.interface_logger as logger
import src.models.model as model_pack
import src.data.dataset as dataset
import src.optimizers.optimizer as optimizers
import src.utils.interface_tensorboard as tensorboard
import src.losses.criterion as losses
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
# nohup tensorboard --logdir=runs --bind_all > /dev/null 2>&1


def main():
    train_tool.setup_seed()
    # Configuration 불러오기
    parser = argparse.ArgumentParser(description='waverdeep - supervised_learning')
    # DISTRIBUTED 사용하기 위해서는 local rank를 argument로 받아야함. 그러면 torch.distributed.launch에서 알아서 해줌
    parser.add_argument("--apex", default=False, type=bool)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--configuration', required=False,
                        default='./config/config_SpeakerClassification_supervised_comp_mobile_training01.json')
    args = parser.parse_args()
    timestamp = train_tool.setup_timestamp()
    config = train_tool.setup_config(args.configuration)
    format_logger = logger.setup_log(save_filename="{}-{}.log".format(config['log_filename'], timestamp))
    train_tool.setup_seed()
    format_logger.info('configurations: {}'.format(config))

    supervised_model = model_pack.load_model(config, config['supervised_model_name'], config['supervised_model_checkpoint'])


    train_loader, train_dataset = dataset.get_dataloader(config=config, mode='train')
    test_loader, test_dataset = dataset.get_dataloader(config=config, mode='test')

    optimizer = optimizers.get_optimizer(model_parameter=supervised_model.parameters(), config=config)
    speaker_dict = train_dataset.speaker_dict

    if config['use_cuda']:
        supervised_model = supervised_model.cuda()

    writer = tensorboard.set_tensorboard_writer("{}-{}".format(config['tensorboard_writer_name'], timestamp))

    # start training
    best_accuracy = 0.0
    num_of_epoch = config['epoch']
    for epoch in range(num_of_epoch):
        epoch = epoch + 1
        format_logger.info("start train ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        train(config, writer, epoch, supervised_model, train_loader, optimizer, format_logger, speaker_dict)
        format_logger.info("start test ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        test_accuracy, test_loss = test(config, writer, epoch, supervised_model,  test_loader, optimizer, format_logger,
                                        speaker_dict)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            train_tool.save_checkpoint(config=config, model=supervised_model, optimizer=optimizer,
                                       loss=test_loss, epoch=best_epoch, format_logger=format_logger, mode="best",
                                       date='{}'.format(timestamp))


def train(config, writer, epoch, supervised_model, train_loader, optimizer, format_logger, speaker_dict):
    supervised_model.train()
    criterion = losses.set_criterion("CrossEntropyLoss")
    total_loss = 0.0
    total_accuracy = 0.0

    format_logger.info("[ {}/{} epoch train result: [ average acc: {}/ average loss: {} ]".format(
        epoch, config['epoch'], total_accuracy, total_loss
    ))

    for batch_idx, (waveform, spectrogram, speaker_id) in enumerate(train_loader):
        targets = train_tool.make_target(speaker_id, speaker_dict)
        if config['use_cuda']:
            data = spectrogram.cuda()
            targets = targets.cuda()
        embeds, output = supervised_model(data)
        loss = criterion(output, targets)
        supervised_model.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = torch.zeros(1)
        _, predicted = torch.max(output.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        accuracy[0] = correct / total

        writer.add_scalar('Loss/train_step', loss, (epoch - 1) * len(train_loader) + batch_idx)
        writer.add_scalar('Accuracy/train_step', accuracy * 100, (epoch - 1) * len(train_loader) + batch_idx)
        total_loss += len(data) * loss
        total_accuracy += len(data) * accuracy

    total_loss /= len(train_loader.dataset)  # average loss
    total_accuracy /= len(train_loader.dataset)  # average acc

    writer.add_scalar('Loss/train', total_loss, (epoch - 1))
    writer.add_scalar('Accuracy/train', total_accuracy * 100, (epoch - 1))


def test(config, writer, epoch, supervised_model, test_loader, optimizer, format_logger, speaker_dict):
    supervised_model.eval()
    criterion = losses.set_criterion("CrossEntropyLoss")
    total_loss = 0.0
    total_accuracy = 0.0

    format_logger.info("[ {}/{} epoch train result: [ average acc: {}/ average loss: {} ]".format(
        epoch, config['epoch'], total_accuracy, total_loss
    ))
    with torch.no_grad():
        for batch_idx, (waveform, spectrogram, speaker_id) in enumerate(test_loader):
            targets = train_tool.make_target(speaker_id, speaker_dict)
            if config['use_cuda']:
                data = spectrogram.cuda()
                targets = targets.cuda()
            embeds, output = supervised_model(data)
            loss = criterion(output, targets)

            accuracy = torch.zeros(1)
            _, predicted = torch.max(output.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            accuracy[0] = correct / total

            writer.add_scalar('Loss/train_step', loss, (epoch - 1) * len(test_loader) + batch_idx)
            writer.add_scalar('Accuracy/train_step', accuracy * 100, (epoch - 1) * len(test_loader) + batch_idx)
            total_loss += len(data) * loss
            total_accuracy += len(data) * accuracy

    total_loss /= len(test_loader.dataset)  # average loss
    total_accuracy /= len(test_loader.dataset)  # average acc

    writer.add_scalar('Loss/train', total_loss, (epoch - 1))
    writer.add_scalar('Accuracy/train', total_accuracy * 100, (epoch - 1))
    return total_accuracy, total_loss


if __name__ == '__main__':
    main()
