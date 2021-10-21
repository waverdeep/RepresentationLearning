import torch
import torch.nn as nn
import argparse
import json
import os
from tqdm import tqdm
import src.data.dataset as dataset
import src.utils.logger as logger
import src.models.model_baseline as model_baseline
import src.optimizers.optimizer_baseline as optimizer_baseline
import src.utils.setup_tensorboard as tensorboard
from apex.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.distributed as distributed
from src.models import model_downstream
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setup_gpu(use, gpu_num):
    if use:
        if torch.cuda.is_available():
            GPU_NUM = gpu_num  # 원하는 GPU 번호 입력
            device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
            torch.cuda.set_device(device)  # change allocation of current GPU


def load_json_config(filename):
    with open(filename, 'r') as configuration:
        config = json.load(configuration)
    return config


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
    parser = argparse.ArgumentParser(
        description='pytorch representation learning downstream task - speaker classification')
    # DISTRIBUTED 사용하기 위해서는 local rank를 argument로 받아야함. 그러면 torch.distributed.launch에서 알아서 해줌
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--configuration', required=False, default='./config/config_spkcls_direct_train01.json')
    args = parser.parse_args()

    # 학습에 필요한 모든 configuration 및 파라미터들을 json으로 불러오기
    config = load_json_config(args.configuration)

    # 로거 생성하기
    format_logger = logger.setup_log(save_filename=config['log_filename'])

    # ddp setting 작업 진행
    args.distributed = False
    setup_distributed_parallel_processing(_distributed=args.distributed,
                                          format_logger=format_logger,
                                          local_rank=args.local_rank)

    # 모델 configuration 출력
    format_logger.info('configurations: {}'.format(config))

    format_logger.info("load train/validation dataset ...")

    train_loader = dataset.get_dataloader_speaker_classification(directory_path=config['dataset']['train_dataset'],
                                                                 speaker_index_file=config['dataset']['train_speaker_list'],
                                                                 audio_window=config['downstream_model']['audio_window'],
                                                                 batch_size=config['model']['batch_size'],
                                                                 num_workers=config['dataset']['num_workers'],
                                                                 shuffle=True,
                                                                 pin_memory=True)

    validation_loader = dataset.get_dataloader_speaker_classification(directory_path=config['dataset']['validation_dataset'],
                                                                      speaker_index_file=config['dataset']['validation_speaker_list'],
                                                                      audio_window=config['downstream_model']['audio_window'],
                                                                      batch_size=config['model']['batch_size'],
                                                                      num_workers=config['dataset']['num_workers'],
                                                                      shuffle=False,
                                                                      pin_memory=True)

    format_logger.info("load model ...")
    # 모델 생성
    pretext_model = model_baseline.CPCType02(args=config,
                                             g_enc_input=config['pretext_model']['g_enc_input'],
                                             g_enc_hidden=config['pretext_model']['g_enc_hidden'],
                                             g_ar_hidden=config['pretext_model']['g_ar_hidden'],
                                             filter_sizes=config['pretext_model']['filter_sizes'],
                                             strides=config['pretext_model']['strides'],
                                             paddings=config['pretext_model']['paddings'])

    downstream_model = model_downstream.SpeakerClassification(256, config['downstream_model']['speaker_num'])
    # GPU 환경 설정
    if config['use_cuda']:
        pretext_model = pretext_model.cuda()
        downstream_model = downstream_model.cuda()

    format_logger.info("load checkpoint ...")
    # pretexet model에 대한 checkpoint 불러오기 (필수로 불러와야 함)
    if config['pretext_model']['load_checkpoint'] != '':
        if config['use_cuda'] == True:
            checkpoint = torch.load(config['pretext_model']['load_checkpoint'], map_location="cuda:0")
            pretext_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            device = torch.device('cpu')
            checkpoint = torch.load(config['pretext_model']['load_checkpoint'], map_location=device)
            pretext_model.load_state_dict(checkpoint['model_state_dict'])

    # downstream task에 대한 checkpoint 불러오기 (있으면)
    if config['checkpoint']['load_checkpoint'] != '':
        if config['use_cuda'] == True:
            checkpoint = torch.load(config['checkpoint']['load_checkpoint'])
            pretext_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            device = torch.device('cpu')
            checkpoint = torch.load(config['checkpoint']['load_checkpoint'], map_location=device)
            pretext_model.load_state_dict(checkpoint['model_state_dict'])

    # DISTRIBUTED 설정
    if args.distributed:
        pretext_model = DDP(pretext_model, delay_allreduce=True)
        downstream_model = DDP(downstream_model, delay_allreduce=True)

    # 텐서보드 셋업
    writer = tensorboard.set_tensorboard_writer(config['tensorboard']['writer_name'])

    # 파라미터 개수 측정
    pretext_model_params = sum(p.numel() for p in pretext_model.parameters() if p.requires_grad)
    downstream_model_params = sum(p.numel() for p in downstream_model.parameters() if p.requires_grad)
    # pretext 모델 정보 출력
    format_logger.info(">>> pretext_model_structure <<<")
    format_logger.info("pretext model parameters: {}".format(pretext_model_params))
    format_logger.info("{}".format(pretext_model_params))
    # downstream 모델 정보 출력
    format_logger.info(">>> downstream_model_structure <<<")
    format_logger.info("downstream model parameters: {}".format(downstream_model_params))
    format_logger.info("{}".format(downstream_model_params))

    # 옵티마이저 생성
    optimizer = optimizer_baseline.get_optimizer(model_parameter=downstream_model.parameters(),
                                                 optimizer_param=config['optimizer'])

    # 학습 코드 작성
    best_accuracy = 0

    num_of_epoch = config['downstream_model']['epoch']
    for epoch in range(num_of_epoch):
        epoch = epoch + 1

        # 학습 함수 시작
        format_logger.info("start training ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        train(config=config,
              writer=writer,
              epoch=epoch,
              pretext_model=pretext_model,
              downstream_model=downstream_model,
              train_loader=train_loader,
              optimizer=optimizer,
              format_logger=format_logger)

        # 검증 함수 시작
        format_logger.info("start validating ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        validation_accuracy, validation_loss = validation(config=config,
                                                          writer=writer,
                                                          epoch=epoch,
                                                          pretext_model=pretext_model,
                                                          downstream_model=downstream_model,
                                                          validation_dataloader=validation_loader,
                                                          format_logger=format_logger)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            save_checkpoint(config=config,
                            model=downstream_model,
                            optimizer=optimizer,
                            loss=validation_loss,
                            epoch=best_epoch,
                            format_logger=format_logger)

        if (epoch+1)/5 == 0:
            save_checkpoint(config=config,
                            model=downstream_model,
                            optimizer=optimizer,
                            loss=validation_loss,
                            epoch=best_epoch,
                            format_logger=format_logger,
                            keyword="-model_epoch{}.pt".format(epoch+1))


    # 텐서보드 종료
    tensorboard.close_tensorboard_writer(writer)


def train(config, writer, epoch, pretext_model, downstream_model, train_loader, optimizer, format_logger):
    pretext_model.eval() # freeze pretrain model parameters
    downstream_model.train()

    criterion = nn.CrossEntropyLoss()

    for batch_idx, [data, target] in enumerate(train_loader):
        # convolution 연산을 위채 1채널 추가
        # format_logger.info("load_data ... ")
        data = data.float().unsqueeze(1)

        # GPU 환경 설정
        if config['use_cuda']:
            data = data.cuda()
            target = target.cuda()
        # gru 모델에 들어간 hidden state 설정하기
        # format_logger.info("pred_model ... ")
        loss, accuracy, z, c = pretext_model(data)
        # print("context size : ", c.size())

        data = c[:, -1, :]
        print("data size : ", data.size())
        # target = target.view((-1,))

        output = downstream_model(data)
        print("output : ", output.size())
        print("target : ", target)
        pred = output.max(1,)[1]  # get the index of the max log-probability
        print("pred : ", pred)
        print("target : ", target.size())
        print("pred : ", pred.size())

        optimizer.zero_grad()
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()


        accuracy = 1. * pred.eq(target.view_as(pred)).sum().item() / len(data)

        # format_logger.info("write_tensorboard ... ")
        # ...학습 중 손실(running loss)을 기록하고
        writer.add_scalar('Loss/train', loss, (epoch - 1) * len(train_loader) + batch_idx)
        writer.add_scalar('Accuracy/train', accuracy * 100, (epoch - 1) * len(train_loader) + batch_idx)


def validation(config, writer, epoch, pretext_model, downstream_model, validation_dataloader, format_logger):
    pretext_model.eval()
    downstream_model.eval()

    criterion = nn.NLLLoss()

    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, [data, target] in enumerate(validation_dataloader):
            # convolution 연산을 위채 1채널 추가
            # format_logger.info("load_data ... ")
            data = data.float().unsqueeze(1)

            # GPU 환경 설정
            if config['use_cuda']:
                data = data.cuda()
                target = target.cuda()
            # gru 모델에 들어간 hidden state 설정하기
            # format_logger.info("pred_model ... ")
            loss, accuracy, z, c = pretext_model(data)

            data = c.contiguous().view((-1, 256))
            target = target.view((-1, 1))

            output = downstream_model(data)

            loss = criterion(output, target)

            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            accuracy = 1. * pred.eq(target.view_as(pred)).sum().item() / len(data)

            # ...검증 중 손실(running loss)을 기록하고
            writer.add_scalar('Loss/validate', loss, (epoch - 1) * len(validation_dataloader) + batch_idx)
            writer.add_scalar('Accuracy/validate', accuracy * 100, (epoch - 1) * len(validation_dataloader) + batch_idx)

            total_loss += loss
            total_accuracy += accuracy

        total_loss /= len(validation_dataloader.dataset)  # average loss
        total_accuracy /= len(validation_dataloader.dataset)  # average acc

        format_logger.info("[ {}/{} epoch validation result: [ average acc: {}/ average loss: {} ]".format(
            epoch, config['downstream_model']['epoch'], total_accuracy, total_loss
        ))
    return total_accuracy, total_loss


def save_checkpoint(config, model, optimizer, loss, epoch, format_logger):
    file_path = os.path.join(config['checkpoint']['save_directory_path'],
                             config['checkpoint']['file_name'] + "-model_best.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "loss": loss}, file_path)
    format_logger.info("saved best checkpoint ... ")


if __name__ == '__main__':
    setup_gpu(use=True, gpu_num=0)
    main()
