import argparse
import json
import random
import numpy as np
import os
from datetime import datetime
import torch.cuda
import src.optimizers.optimizer as optimizers
import src.data.dataset as dataset
import src.models.model as model_pack
import src.utils.interface_logger as logger
import src.utils.interface_tensorboard as tensorboard
import src.utils.interface_plot as plots
import src.utils.interface_train_tool as train_tool
from apex.parallel import DistributedDataParallel as DDP
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main():
    # Configuration 불러오기
    parser = argparse.ArgumentParser(description='waverdeep - representation learning')
    # DISTRIBUTED 사용하기 위해서는 local rank를 argument로 받아야함. 그러면 torch.distributed.launch에서 알아서 해줌
    parser.add_argument("--apex", default=False, type=bool)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--configuration', required=False,
                        default='./config/config_pretext-GCPC-librispeech100-training01-batch256.json')
    args = parser.parse_args()

    train_tool.setup_seed(random_seed=777)
    now = train_tool.setup_timestamp()

    # read configuration file
    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    # create logger
    format_logger = logger.setup_log(save_filename="{}-{}.log".format(config['log_filename'], now))

    # gpu check
    format_logger.info("GPU: {}".format(torch.cuda.is_available()))
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        format_logger.info("APEX DISTRIBUTED: {}".format(args.distributed))

    # setup distributed training
    if args.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(args.local_rank)

        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.benchmark = True

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

    # if ditributed training available:
    if args.distributed:
        model = DDP(model, delay_allreduce=True)

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
    best_accuracy = 0.0
    num_of_epoch = config['epoch']
    for epoch in range(num_of_epoch):
        epoch = epoch + 1
        format_logger.info("start train ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        train(config, writer, epoch, model, train_loader, optimizer, format_logger)
        format_logger.info("start test ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        test_accuracy, test_loss = test(config, writer, epoch, model, test_loader, format_logger)
        # speaker_tsne(config, model, train_dataset, epoch, writer)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            save_checkpoint(config=config, model=model, optimizer=optimizer,
                            loss=test_loss, epoch=best_epoch, format_logger=format_logger, mode="best",
                            date='{}'.format(now))

    tensorboard.close_tensorboard_writer(writer)


def train(config, writer, epoch, model, train_loader, optimizer, format_logger):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    with torch.autograd.set_detect_anomaly(True):
        for batch_idx, (waveform, filename, speaker_id) in enumerate(train_loader):
            if config['use_cuda']:
                data = waveform.cuda()
            loss, accuracy, z, c = model(data)
            # optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train_step', loss, (epoch - 1) * len(train_loader) + batch_idx)
            writer.add_scalar('Accuracy/train_step', accuracy * 100, (epoch - 1) * len(train_loader) + batch_idx)
            total_loss += len(data) * loss
            total_accuracy += len(data) * accuracy

    total_loss /= len(train_loader.dataset)  # average loss
    total_accuracy /= len(train_loader.dataset)  # average acc

    writer.add_scalar('Loss/train', total_loss, (epoch - 1))
    writer.add_scalar('Accuracy/train', total_accuracy * 100, (epoch - 1))

    conv = 0
    conv_trans = 0
    for idx, layer in enumerate(model.modules()):
        if isinstance(layer, torch.nn.Conv1d):
            writer.add_histogram("Conv/weights-{}".format(conv), layer.weight,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            writer.add_histogram("Conv/bias-{}".format(conv), layer.bias,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            conv += 1

        if isinstance(layer, torch.nn.ConvTranspose1d):
            writer.add_histogram("ConvTrans/weights-{}".format(conv_trans), layer.weight,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            writer.add_histogram("ConvTrans/bias-{}".format(conv_trans), layer.bias,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            conv_trans += 1

        if isinstance(layer, torch.nn.GRU):
            writer.add_histogram("GRU/weight_ih_l0",
                                 layer.weight_ih_l0, global_step=(epoch - 1) * len(train_loader) + batch_idx)
            writer.add_histogram("GRU/weight_hh_l0",
                                 layer.weight_hh_l0, global_step=(epoch - 1) * len(train_loader) + batch_idx)
            writer.add_histogram("GRU/bias_ih_l0",
                                 layer.bias_ih_l0, global_step=(epoch - 1) * len(train_loader) + batch_idx)
            writer.add_histogram("GRU/bias_hh_l0",
                                 layer.bias_hh_l0, global_step=(epoch - 1) * len(train_loader) + batch_idx)


def test(config, writer, epoch, model, test_loader, format_logger):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, (waveform, filename, speaker_id) in enumerate(test_loader):
            if config['use_cuda']:
                data = waveform.cuda()
            loss, accuracy, z, c = model(data)

            writer.add_scalar('Loss/test_step', loss, (epoch - 1) * len(test_loader) + batch_idx)
            writer.add_scalar('Accuracy/test_step', accuracy * 100, (epoch - 1) * len(test_loader) + batch_idx)

            total_loss += len(data) * loss
            total_accuracy += len(data) * accuracy

        total_loss /= len(test_loader.dataset)  # average loss
        total_accuracy /= len(test_loader.dataset)  # average acc

        writer.add_scalar('Loss/test', total_loss, (epoch - 1))
        writer.add_scalar('Accuracy/test', total_accuracy * 100, (epoch - 1))

        format_logger.info("[ {}/{} epoch validation result: [ average acc: {}/ average loss: {} ]".format(
            epoch, config['epoch'], total_accuracy, total_loss
        ))

    return total_accuracy, total_loss


def save_checkpoint(config, model, optimizer, loss, epoch, format_logger, mode="best", date=""):
    if mode == "best":
        file_path = os.path.join(config['checkpoint_save_directory_path'],
                                 config['checkpoint_file_name'] + "-model-best-{}.pt".format(date))

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "loss": loss}, file_path)
    format_logger.info("saved best checkpoint ... ")


def speaker_tsne(config, model, dataset, epoch, writer):
    tsne_cluster = 10
    batch_size = config['batch_size']
    audio_window = config['audio_window']
    input_size = (batch_size, 1, audio_window)

    model.eval()
    with torch.no_grad():
        latent_rep_size, latent_rep_len = model.get_latent_size(input_size)
        features = torch.zeros(tsne_cluster, batch_size, latent_rep_size * latent_rep_len).cuda()
        labels = torch.zeros(tsne_cluster, batch_size).cuda()

        for index, speaker_idx in enumerate(dataset.speaker_align):
            if index == tsne_cluster:
                break

            input_data = dataset.get_audio_by_speaker(speaker_idx, batch_size=batch_size)
            input_data = input_data.cuda()
            z, c = model.get_latent_representations(input_data)

            z_representation = z.permute(0, 2, 1)
            c_representation = c.permute(0, 2, 1)

            features[index, :, :] = c_representation.reshape((batch_size, -1))
            labels[index, :] = index

    features = features.reshape(features.size(0) * features.size(1), -1).cpu()
    labels = labels.reshape(-1, 1).cpu().numpy()

    embedding = plots.tsne(config, features)
    figure = plots.plot_tsne(config, embedding, labels, epoch)

    # add to TensorBoard
    writer.add_embedding(features, metadata=labels, global_step=epoch)
    writer.add_figure("TSNE", figure, global_step=epoch)
    writer.flush()


if __name__ == '__main__':
    main()
