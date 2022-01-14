import argparse
import os
import torch
import json
import src.utils.interface_logger as logger
import src.utils.interface_train_tool as train_tool
import src.utils.interface_tensorboard as tensorboard
import src.data.dataset as dataset
import src.models.model as model_pack
import src.optimizers.optimizer as optimizers
import src.utils.interface_audio_augmentation as audio_augmentation
import src.optimizers.ExponentialMovingAverage as ema
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    # Configuration 불러오기
    parser = argparse.ArgumentParser(description='waverdeep - byol_audio implementation')
    # DISTRIBUTED 사용하기 위해서는 local rank를 argument로 받아야함. 그러면 torch.distributed.launch에서 알아서 해줌
    parser.add_argument("--apex", default=False, type=bool)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--configuration', required=False,
                        default='./config/config_pretext-BYOLA-urbansound-training01-batch256.json')
    args = parser.parse_args()

    # train_tool.setup_seed(random_seed=777)
    now = train_tool.setup_timestamp()

    # read configuration file
    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    # create logger
    format_logger = logger.setup_log(save_filename="{}-{}.log".format(config['log_filename'], now))

    # gpu check
    format_logger.info("GPU: {}".format(torch.cuda.is_available()))

    # print configuration 출력
    format_logger.info('configurations: {}'.format(config))
    format_logger.info('load train/test dataset ...')

    # setup train/test dataloader
    train_loader, train_dataset = dataset.get_dataloader(config=config, mode='train')
    test_loader, _ = dataset.get_dataloader(config=config, mode='test')

    # load model
    format_logger.info("load_model ...")
    model = model_pack.load_model(config, model_name=config['pretext_model_name'])

    # if gpu available: load gpu
    if config['use_cuda']:
        model = model.cuda()

    # setup optimizer
    optimizer = optimizers.get_optimizer(model_parameter=model.parameters(),
                                         config=config)

    # setup tensorboard
    writer = tensorboard.set_tensorboard_writer(
        "{}-{}".format(config['tensorboard_writer_name'], now)
    )

    # print model information
    format_logger.info(">>> model_structure <<<")
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    format_logger.info("model parameters: {}".format(model_params))
    format_logger.info("{}".format(model))

    # start training ....
    best_loss = 0.0
    num_of_epoch = config['epoch']
    for epoch in range(num_of_epoch):
        epoch = epoch + 1
        format_logger.info("start train ... [ {}/{} epoch - {} iter ]".format(epoch, num_of_epoch, len(train_loader)))
        train(config, writer, epoch, model, train_loader, optimizer, format_logger)
        format_logger.info("start test ... [ {}/{} epoch - {} iter ]".format(epoch, num_of_epoch, len(test_loader)))
        test_loss = test(config, writer, epoch, model, test_loader, format_logger)

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            train_tool.save_checkpoint(config=config, model=model, optimizer=optimizer,
                                       loss=test_loss, epoch=best_epoch, format_logger=format_logger, mode="best",
                                       date='{}'.format(now))

    tensorboard.close_tensorboard_writer(writer)


def train(config, writer, epoch, model, train_loader, optimizer, format_logger):
    model.train()
    total_loss = 0.0
    post_norm = audio_augmentation.NormalizeBatch()
    target_ema = ema.EMA(config['ema_decay'])
    for batch_idx, (waveform, filename, speaker_id) in enumerate(train_loader):
        bs = int(waveform[0].shape[0])
        waveform = torch.cat(waveform)  # [(B,1,F,T), (B,1,F,T)] -> (2*B,1,F,T)
        waveform = post_norm(waveform)

        if config['use_cuda']:
            data = waveform.cuda()

        _, loss = model(data[:bs], data[bs:])
        model.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train_step', loss, (epoch - 1) * len(train_loader) + batch_idx)
        total_loss += len(data) * loss

    total_loss /= len(train_loader.dataset)  # average loss

    writer.add_scalar('Loss/train', total_loss, (epoch - 1))

    conv1 = 0
    conv2 = 0
    for idx, layer in enumerate(model.modules()):
        if isinstance(layer, torch.nn.Conv1d):
            writer.add_histogram("Conv1/weights-{}".format(conv1), layer.weight,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            writer.add_histogram("Conv1/bias-{}".format(conv1), layer.bias,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            conv1 += 1

        if isinstance(layer, torch.nn.Conv2d):
            writer.add_histogram("Conv2/weights-{}".format(conv2), layer.weight,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            writer.add_histogram("Conv2/bias-{}".format(conv2), layer.bias,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            conv2 += 1

    # exponential moving average 적용
    ema.update_moving_average(target_ema, model.target_encoder, model.online_encoder)
    ema.update_moving_average(target_ema, model.target_projector, model.online_projector)


def test(config, writer, epoch, model, test_loader, format_logger):
    model.eval()
    total_loss = 0.0
    post_norm = audio_augmentation.NormalizeBatch()
    with torch.no_grad():
        for batch_idx, (waveform, filename, speaker_id) in enumerate(test_loader):
            bs = int(waveform[0].shape[0])
            waveform = torch.cat(waveform)  # [(B,1,F,T), (B,1,F,T)] -> (2*B,1,F,T)
            waveform = post_norm(waveform)

            if config['use_cuda']:
                data = waveform.cuda()

            _, loss = model(data[:bs], data[bs:])

            writer.add_scalar('Loss/test_step', loss, (epoch - 1) * len(test_loader) + batch_idx)
            total_loss += len(data) * loss

        total_loss /= len(test_loader.dataset)  # average loss
        writer.add_scalar('Loss/test', total_loss, (epoch - 1))
        return total_loss


if __name__ == '__main__':
    main()
