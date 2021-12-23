import train as train_tool
import src.utils.interface_logger as logger
import src.data as dataset
import src.models.model as model_pack
import src.optimizers.optimizer as optimizers
import src.utils.interface_tensorboard as tensorboard
import src.losses.criterion as criterions
import torch


def main():
    args = train_tool.setup_argparse()
    timestamp = train_tool.setup_timestamp()
    config = train_tool.setup_config(args.configuration)
    format_logger = logger.setup_log(save_filename="{}-{}.log".format(config['log_filename'], timestamp))
    train_tool.setup_seed()
    format_logger.info('configurations: {}'.format(config))

    # load dataset
    train_loader, train_dataset = dataset.get_dataloader(config=config, mode='train')
    test_loader, test_dataset = dataset.get_dataloader(config=config, mode='test')

    # load pretext model
    pretext_model = model_pack.load_model(config, config['pretext_model_name'], config['pretext_checkpoint'])
    downstream_model = model_pack.load_model(config, config['downstream_model_name'])

    # setup optimizer
    optimizer = optimizers.get_optimizer(model_parameter=downstream_model.parameters(),
                                         config=config)

    # if gpu available: load gpu
    if config['use_cuda']:
        pretext_model = pretext_model.cuda()
        downstream_model = downstream_model.cuda()

    writer = tensorboard.set_tensorboard_writer("{}-{}".format(config['tensorboard_writer_name'], timestamp))

    train_tool.print_model_description("pretext")
    train_tool.print_model_description("downstream")

    # load target
    phone_dict_dataset = []

    # start training
    best_accuracy = 0.0
    num_of_epoch = config['epoch']
    for epoch in range(num_of_epoch):
        epoch = epoch + 1
        format_logger.info("start train ... [ {}/{} epoch ]".format(epoch, num_of_epoch))

        format_logger.info("start test ... [ {}/{} epoch ]".format(epoch, num_of_epoch))


def train(config, writer, epoch, pretext_model, downstream_model, train_loader):
    pretext_model.eval()
    downstream_model.train()
    criterion = criterions.set_criterion(config['criterion'])
    total_loss = 0.0
    total_accuracy = 0.0
    for batch_idx, (waveform, filename, speaker_id) in enumerate(train_loader):
        if config['use_cuda']:
            data = waveform.cuda()
        with torch.no_grad():
            loss, accuracy, z, c = pretext_model(data)
    c = c.detach()



def test():
    pass





