import torch
import argparse
import os
import src.data.dataset as dataset
import src.utils.interface_logger as logger
import src.utils.interface_file_io as file_io
import src.models.model_baseline as model_baseline
import src.optimizers.optimizer as optimizer_baseline
import src.utils.interface_tensorboard as tensorboard
from apex.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.distributed as distributed
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setup_gpu(use, gpu_num):
    if use:
        if torch.cuda.is_available():
            GPU_NUM = gpu_num  # 원하는 GPU 번호 입력
            device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
            torch.cuda.set_device(device)  # change allocation of current GPU


def setup_distributed_parallel_processing(_distributed, format_logger, local_rank):
    # DISTRIBUTED 사용하기 위해서 아래 작업을 해야하는데, WORLD_SIZE도 알아서 해줌
    if 'WORLD_SIZE' in os.environ:
        _distributed = int(os.environ['WORLD_SIZE']) > 1
        format_logger.info("APEX DISTRIBUTED: {}".format(_distributed))

    # DISTRIBUTED 할 수 있도록 .
    if _distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(local_rank)

        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    torch.backends.cudnn.benchmark = True


def main():
    # CONFIGURATION 파라미터를 argparse로 받아오기
    parser = argparse.ArgumentParser(description='pytorch representation learning type02')
    # DISTRIBUTED 사용하기 위해서는 local rank를 argument로 받아야함. 그러면 torch.distributed.launch에서 알아서 해줌
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--configuration', required=False, default='./config/config_type02_direct_train50(norm_torch).json')
    args = parser.parse_args()

    # 학습에 필요한 모든 configuration 및 파라미터들을 json으로 불러오기
    config = file_io.load_json_config(args.configuration)

    # 로거 생성하기
    format_logger = logger.setup_log(save_filename=config['log_filename'])

    # DISTRIBUTED 사용하기 위해서 아래 작업을 해야하는데, WORLD_SIZE도 알아서 해줌
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        format_logger.info("APEX DISTRIBUTED: {}".format(args.distributed))

    # DISTRIBUTED 할 수 있도록 .
    if args.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(args.local_rank)

        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    torch.backends.cudnn.benchmark = True

    # 모델 configuration 출력
    format_logger.info('configurations: {}'.format(config))
    format_logger.info("load train/validation dataset ...")
    # 학습 데이터로더 생성
    train_loader = dataset.get_dataloader_type_direct(directory_path=config['dataset']['train_dataset'],
                                                      audio_window=config['model']['audio_window'],
                                                      batch_size=config['model']['batch_size'],
                                                      num_workers=config['dataset']['num_workers'],
                                                      shuffle=True, pin_memory=True)
    # 검증 데이터로더 생성
    validation_loader = dataset.get_dataloader_type_direct(directory_path=config['dataset']['validation_dataset'],
                                                           audio_window=config['model']['audio_window'],
                                                           batch_size=config['model']['batch_size'],
                                                           num_workers=config['dataset']['num_workers'],
                                                           shuffle=False, pin_memory=True)

    format_logger.info("load model ...")
    # 모델 생성
    model = model_baseline.CPCType02(args=config,
                                     g_enc_input=config['model']['g_enc_input'],
                                     g_enc_hidden=config['model']['g_enc_hidden'],
                                     g_ar_hidden=config['model']['g_ar_hidden'],
                                     filter_sizes=config['model']['filter_sizes'],
                                     strides=config['model']['strides'],
                                     paddings=config['model']['paddings'])
    # GPU 환경 설정
    if config['use_cuda']:
        format_logger.info("setup cuda ... ")
        model = model.cuda()

    format_logger.info("load checkpoint ...")
    if config['checkpoint']['load_checkpoint'] != '':
        if config['use_cuda'] == True:
            checkpoint = torch.load(config['checkpoint']['load_checkpoint'])
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            device = torch.device('cpu')
            checkpoint = torch.load(config['checkpoint']['load_checkpoint'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

    # DISTRIBUTED 설정
    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    # 텐서보드 셋업
    format_logger.info("setup tensorboard ...")
    writer = tensorboard.set_tensorboard_writer(config['tensorboard']['writer_name'])

    # 모델 정보 출력
    format_logger.info(">>> model_structure <<<")
    # 파라미터 개수 측정
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    format_logger.info("model parameters: {}".format(model_params))
    format_logger.info("{}".format(model))

    # 옵티마이저 생성
    optimizer = optimizer_baseline.get_optimizer(model_parameter=model.parameters(),
                                                 optimizer_param=config['optimizer'])

    # 학습 코드 작성
    best_accuracy = 0
    global_step = 0
    print_step = 100
    validation_step = 0

    num_of_epoch = config['model']['epoch']
    for epoch in range(num_of_epoch):
        # 학습 함수 시작
        format_logger.info("start training ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        train(config=config,
              writer=writer,
              epoch=epoch,
              model=model,
              train_loader=train_loader,
              optimizer=optimizer,
              format_logger=format_logger,
              global_step=global_step,
              print_step=print_step)
        # 검증 함수 시작
        format_logger.info("start validating ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        validation_accuracy, validation_loss = validation(config=config,
                                                          writer=writer,
                                                          epoch=epoch,
                                                          model=model,
                                                          validation_loader=validation_loader,
                                                          format_logger=format_logger,
                                                          global_step=validation_step,
                                                          print_step=print_step)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            save_checkpoint(config=config,
                            model=model,
                            optimizer=optimizer,
                            loss=validation_loss,
                            epoch=best_epoch,
                            format_logger=format_logger,
                            keyword="-model_best.pt")
        if (epoch + 1) % 10 == 0:
            save_checkpoint(config=config,
                            model=model,
                            optimizer=optimizer,
                            loss=validation_loss,
                            epoch=best_epoch,
                            format_logger=format_logger,
                            keyword="-model_epoch{}.pt".format(epoch+1))

    # # speaker classification을 이어서 작업해보아도 좋을 듯
    # load_speaker_classification(model, format_logger, writer)

    # 텐서보드 종료
    tensorboard.close_tensorboard_writer(writer)


def train(config, writer, epoch, model, train_loader, optimizer, format_logger, global_step, print_step):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for batch_idx, data in enumerate(train_loader):
        # GPU 환경 설정
        if config['use_cuda']:
            data = data.cuda()
        loss, accuracy, _, _ = model(data)
        # 옵티마이저 초기화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % print_step == 0:
            format_logger.info(
                "[Epoch {}/{}] train step {:04d}/{:04d} \t"
                "Loss = {:.4f} Accuracy = {:.4f}".format(
                    epoch, config['model']['epoch'],
                    batch_idx, len(train_loader),
                    loss,
                    accuracy,
                )
            )
        # ...학습 중 손실(running loss)을 기록하고
        writer.add_scalar('Loss/train_step', loss, global_step)
        writer.add_scalar('Accuracy/train_step', accuracy, global_step)

        total_loss += loss
        total_accuracy += accuracy
        global_step += 1

    total_loss /= len(train_loader)  # average loss
    total_accuracy /= len(train_loader)  # average accuracy

    writer.add_scalar('Loss/train_epoch', total_loss, epoch)
    writer.add_scalar('Accuracy/train_epoch', total_accuracy, epoch)

    format_logger.info(
        "[Epoch {}/{}] Loss {:.4f} Accuracy {:.4f}".format(
            epoch, config['model']['epoch'],
            total_loss, total_accuracy
        )
    )

    # 학습 중 각각 레이어의 weight 분포를 알아보기 위해
    conv = 0
    for idx, layer in enumerate(model.modules()):
        if isinstance(layer, torch.nn.Conv1d):
            writer.add_histogram("Conv/weights-{}".format(conv), layer.weight, global_step=(epoch-1) * len(train_loader) + batch_idx)
            writer.add_histogram("Conv/bias-{}".format(conv), layer.bias, global_step=(epoch-1) * len(train_loader) + batch_idx)
            conv += 1
        if isinstance(layer, torch.nn.GRU):
            writer.add_histogram("GRU/weight_ih_l0",
                                 layer.weight_ih_l0, global_step=(epoch-1) * len(train_loader) + batch_idx)
            writer.add_histogram("GRU/weight_hh_l0",
                                 layer.weight_hh_l0, global_step=(epoch-1) * len(train_loader) + batch_idx)
            writer.add_histogram("GRU/bias_ih_l0",
                                 layer.bias_ih_l0, global_step=(epoch - 1) * len(train_loader) + batch_idx)
            writer.add_histogram("GRU/bias_hh_l0",
                                 layer.bias_hh_l0, global_step=(epoch - 1) * len(train_loader) + batch_idx)


def validation(config, writer, epoch, model, validation_loader, format_logger, global_step, print_step):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            # GPU 환경 설정
            if config['use_cuda']:
                data = data.cuda()
            loss, accuracy, _, _ = model(data)

            # ...검증 중 손실(running loss)을 기록하고
            writer.add_scalar('Loss/valid_step', loss, global_step)
            writer.add_scalar('Accuracy/valid_step', accuracy, global_step)

            total_loss += loss
            total_accuracy += accuracy
            global_step += 1
            if batch_idx % print_step == 0:
                format_logger.info(
                    "[Epoch {}/{}] train step {:04d}/{:04d} \t "
                    "Loss = {:.4f} Accuracy = {:.4f}".format(
                        epoch, config['model']['epoch'],
                        batch_idx, len(validation_loader),
                        loss,
                        accuracy,
                    )
                )

        total_loss /= len(validation_loader)  # average loss
        total_accuracy /= len(validation_loader)  # average acc

        writer.add_scalar('Loss/valid_epoch', total_loss, epoch)
        writer.add_scalar('Accuracy/valid_epoch', total_accuracy, epoch)

        format_logger.info(
            "[Epoch {}/{}] Loss {:.4f} Accuracy {:.4f}".format(
                epoch, config['model']['epoch'],
                total_loss, total_accuracy
            )
        )
    return total_accuracy, total_loss


def save_checkpoint(config, model, optimizer, loss, epoch, format_logger, keyword):
    file_path = os.path.join(config['checkpoint']['save_directory_path'],
                             config['checkpoint']['file_name'] + keyword)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "loss": loss}, file_path)
    format_logger.info("saved best checkpoint ... ")


if __name__ == '__main__':
    setup_gpu(False, 2)
    main()
