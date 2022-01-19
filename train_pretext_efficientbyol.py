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
import src.losses.criterion_metrics as metrics
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main():
    # Configuration 불러오기
    parser = argparse.ArgumentParser(description='waverdeep - waveBYOL proposed')
    parser.add_argument('--configuration', required=False,
                        default='./config/config_pretext-EfficientBYOL-librispeech100-efficientb7aug-batch64.json')
    args = parser.parse_args()
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
        train(config, writer, epoch, model, train_loader, optimizer)
        format_logger.info("start test ... [ {}/{} epoch - {} iter ]".format(epoch, num_of_epoch, len(test_loader)))
        test_loss = test(config, writer, epoch, model, test_loader)

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            train_tool.save_checkpoint(config=config, model=model, optimizer=optimizer,
                                       loss=test_loss, epoch=best_epoch, format_logger=format_logger, mode="best",
                                       date='{}'.format(now))

    tensorboard.close_tensorboard_writer(writer)


def train(config, writer, epoch, model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    target_ema = ema.EMA(config['ema_decay'])
    # tensorboard.add_dataset_figure(writer, train_loader, "Train", epoch)
    for batch_idx, (waveform01, waveform02, waveform03, filename, speaker_id) in enumerate(train_loader):
        if config['use_cuda']:
            data01 = waveform01.cuda()
            data02 = waveform02.cuda()
            data03 = waveform03.cuda()
        online01_pre, online02_pre, online01_rep, online02_rep, \
        target01_pre, target02_pre, target01_rep, target02_rep, \
        loss = model(data01, data02, data03)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train_step', loss, (epoch - 1) * len(train_loader) + batch_idx)
        total_loss += len(data01) * loss

        if batch_idx % 50 == 0:
            online01_output = online01_pre.detach()
            online02_output = online02_pre.detach()
            target01_output = target01_pre.detach()
            target02_output = target02_pre.detach()
            online01_output = online01_output[0].cpu().numpy()
            online02_output = online02_output[0].cpu().numpy()
            target01_output = target01_output[0].cpu().numpy()
            target02_output = target02_output[0].cpu().numpy()

            tensorboard.add_byol_latent_heatmap(writer, online01_output, target02_output, "TrainLatentPreVector",
                                                "online01_vs_target02", (epoch - 1) * len(train_loader) + batch_idx)
            tensorboard.add_byol_latent_heatmap(writer, online02_output, target01_output, "TrainLatentPreVector",
                                                "online02_vs_target01", (epoch - 1) * len(train_loader) + batch_idx)

            online01_output_rep = online01_rep.detach()
            online02_output_rep = online02_rep.detach()
            target01_output_rep = target01_rep.detach()
            target02_output_rep = target02_rep.detach()
            online01_output_rep = online01_output_rep[0].squeeze().cpu().numpy()
            online02_output_rep = online02_output_rep[0].squeeze().cpu().numpy()
            target01_output_rep = target01_output_rep[0].squeeze().cpu().numpy()
            target02_output_rep = target02_output_rep[0].squeeze().cpu().numpy()

            tensorboard.add_byol_latent_heatmap(writer, online01_output_rep, target02_output_rep,
                                                "TrainLatentEncoderVector",
                                                "online01_vs_target02", (epoch - 1) * len(train_loader) + batch_idx)
            tensorboard.add_byol_latent_heatmap(writer, online02_output_rep, target01_output_rep,
                                                "TrainLatentEncoderVector",
                                                "online02_vs_target01", (epoch - 1) * len(train_loader) + batch_idx)

    total_loss /= len(train_loader.dataset)  # average loss
    writer.add_scalar('Loss/train', total_loss, (epoch - 1))

    ema.update_moving_average(target_ema, model.target_pre_network, model.online_pre_network)
    ema.update_moving_average(target_ema, model.target_encoder_network, model.online_encoder_network)
    ema.update_moving_average(target_ema, model.target_projector_network, model.online_projector_network)
    ema.update_moving_average(target_ema, model.modest_pre_network, model.target_pre_network)
    ema.update_moving_average(target_ema, model.modest_encoder_network, model.target_encoder_network)
    ema.update_moving_average(target_ema, model.modest_projector_network, model.target_projector_network)


    conv1d = 0
    conv2d = 0
    for idx, layer in enumerate(model.modules()):
        if isinstance(layer, torch.nn.Conv1d):
            writer.add_histogram("Conv1d/weights-{}".format(conv1d), layer.weight,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            writer.add_histogram("Conv1d/bias-{}".format(conv1d), layer.bias,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            conv1d += 1

        if isinstance(layer, torch.nn.Conv2d):
            writer.add_histogram("Conv2d/weights-{}".format(conv2d), layer.weight,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            # writer.add_histogram("Conv2/bias-{}".format(conv2), layer.bias,
            #                      global_step=(epoch - 1) * len(train_loader) + batch_idx)
            conv2d += 1


def test(config, writer, epoch, model, test_loader):
    model.eval()
    total_loss = 0.0
    # tensorboard.add_dataset_figure(writer, test_loader, "Test", epoch)
    with torch.no_grad():
        for batch_idx, (waveform01, waveform02, waveform03, filename, speaker_id) in enumerate(test_loader):
            if config['use_cuda']:
                data01 = waveform01.cuda()
                data02 = waveform02.cuda()
                data03 = waveform03.cuda()
            online01_pre, online02_pre, online01_rep, online02_rep, \
            target01_pre, target02_pre, target01_rep, target02_rep, \
            loss = model(data01, data02, data03)
            writer.add_scalar('Loss/test_step', loss, (epoch - 1) * len(test_loader) + batch_idx)
            total_loss += len(data01) * loss

            if batch_idx % 50 == 0:
                online01_output = online01_pre.detach()
                online02_output = online02_pre.detach()
                target01_output = target01_pre.detach()
                target02_output = target02_pre.detach()
                online01_output = online01_output[0].cpu().numpy()
                online02_output = online02_output[0].cpu().numpy()
                target01_output = target01_output[0].cpu().numpy()
                target02_output = target02_output[0].cpu().numpy()

                tensorboard.add_byol_latent_heatmap(writer, online01_output, target02_output, "TestLatentPreVector",
                                                    "online01_vs_target02", (epoch - 1) * len(test_loader) + batch_idx)
                tensorboard.add_byol_latent_heatmap(writer, online02_output, target01_output, "TestLatentPreVector",
                                                    "online02_vs_target01", (epoch - 1) * len(test_loader) + batch_idx)

                online01_output_rep = online01_rep.detach()
                online02_output_rep = online02_rep.detach()
                target01_output_rep = target01_rep.detach()
                target02_output_rep = target02_rep.detach()
                online01_output_rep = online01_output_rep[0].squeeze().cpu().numpy()
                online02_output_rep = online02_output_rep[0].squeeze().cpu().numpy()
                target01_output_rep = target01_output_rep[0].squeeze().cpu().numpy()
                target02_output_rep = target02_output_rep[0].squeeze().cpu().numpy()

                tensorboard.add_byol_latent_heatmap(writer, online01_output_rep, target02_output_rep,
                                                    "TestLatentEncoderVector",
                                                    "online01_vs_target02", (epoch - 1) * len(test_loader) + batch_idx)
                tensorboard.add_byol_latent_heatmap(writer, online02_output_rep, target01_output_rep,
                                                    "TestLatentEncoderVector",
                                                    "online02_vs_target01", (epoch - 1) * len(test_loader) + batch_idx)

        total_loss /= len(test_loader.dataset)  # average loss
        writer.add_scalar('Loss/test', total_loss, (epoch - 1))
        return total_loss


if __name__ == '__main__':
    main()
