import torch
import argparse
import os
import src.data.dataset as dataset
import src.utils.interface_logger as logger
import src.utils.interface_file_io as file_io
import src.models.model_baseline as model_baseline
import src.optimizers.optimizer as optimizer_baseline
import src.utils.interface_tensorboard as tensorboard
import src.losses.criterion as criterion
from apex.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.distributed as distributed
from src.models import model_downstream
from sklearn.metrics import accuracy_score
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
    parser = argparse.ArgumentParser(
        description='pytorch representation learning downstream task - speaker classification')
    # DISTRIBUTED 사용하기 위해서는 local rank를 argument로 받아야함. 그러면 torch.distributed.launch에서 알아서 해줌
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--configuration', required=False, default='./config/config_spkcls_direct_train02.json')
    args = parser.parse_args()

    # 학습에 필요한 모든 configuration 및 파라미터들을 json으로 불러오기
    config = file_io.load_json_config(args.configuration)

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
            pretext_model.load_state_dict(checkpoint['model_module_state_dict'])
        else:
            device = torch.device('cpu')
            checkpoint = torch.load(config['pretext_model']['load_checkpoint'], map_location=device)
            pretext_model.load_state_dict(checkpoint['model_state_dict'])

    # downstream task에 대한 checkpoint 불러오기 (있으면)
    if config['checkpoint']['load_checkpoint'] != '':
        if config['use_cuda'] == True:
            checkpoint = torch.load(config['checkpoint']['load_checkpoint'], map_location="cuda:0")
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
    # loss 생성
    loss_function = criterion.set_criterion("NLLLoss")

    # 학습 코드 작성
    best_accuracy = 0
    global_step = 0
    print_step = 100

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
              loss_function=loss_function,
              format_logger=format_logger,
              global_step=global_step,
              print_step=print_step)

        # 검증 함수 시작
        format_logger.info("start validating ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        validation_accuracy, validation_loss = validation(config=config,
                                                          writer=writer,
                                                          epoch=epoch,
                                                          pretext_model=pretext_model,
                                                          downstream_model=downstream_model,
                                                          loss_function=loss_function,
                                                          validation_loader=validation_loader,
                                                          format_logger=format_logger,
                                                          global_step=global_step,
                                                          print_step=print_step)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            save_checkpoint(config=config,
                            model=downstream_model,
                            optimizer=optimizer,
                            loss=validation_loss,
                            epoch=best_epoch,
                            format_logger=format_logger)

        if (epoch + 1 ) % 10 == 0:
            save_checkpoint(config=config,
                            model=downstream_model,
                            optimizer=optimizer,
                            loss=validation_loss,
                            epoch=best_epoch,
                            format_logger=format_logger,
                            keyword="-model_epoch{}.pt".format(epoch+1))


    # 텐서보드 종료
    tensorboard.close_tensorboard_writer(writer)


def train(config, writer, epoch, pretext_model, downstream_model, train_loader, optimizer,
          loss_function, format_logger, global_step, print_step):
    pretext_model.eval() # freeze pretrain model parameters
    downstream_model.train()

    total_loss = 0.0
    total_accuracy = 0.0

    for batch_idx, [data, target] in enumerate(train_loader):
        # GPU 환경 설정
        if config['use_cuda']:
            data = data.cuda()
            target = target.cuda()
        with torch.no_grad():
            p_loss, p_accuracy, z, c = pretext_model(data)
        # z -> latent vector
        # c -> context vector : 나는 가장 마지막에 추출된 context vector를 사용해서 classification을
        #                       진행할 예정
        _data = c[:, -1, :]
        output = downstream_model(_data)
        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = output.max(1)[1].cpu()
        target = target.cpu()
        accuracy = accuracy_score(target, pred)

        writer.add_scalar('Loss/spcls_train_step', loss, global_step)
        writer.add_scalar('Accuracy/spcls_train_step', accuracy, global_step)

        total_loss += loss
        total_accuracy += accuracy
        global_step += 1
        if batch_idx % print_step == 0:
            format_logger.info(
                "[Epoch {}/{}] train step {:04d}/{:04d} \t"
                "Loss = {:.4f} Accuracy = {:.4f}".format(
                    epoch, config['downstream_model']['epoch'],
                    batch_idx, len(train_loader),
                    loss,
                    accuracy,
                )
            )

    total_loss /= len(train_loader)  # average loss
    total_accuracy /= len(train_loader)  # average acc

    writer.add_scalar('Loss/spcls_train_epoch', total_loss, epoch)
    writer.add_scalar('Accuracy/spcls_train_epoch', total_accuracy, epoch)

    format_logger.info(
        "[Epoch {}/{}] Loss {:.4f} Accuracy {:.4f}".format(
            epoch, config['downstream_model']['epoch'],
            total_loss, total_accuracy
        )
    )

    linear = 0
    for idx, layer in enumerate(downstream_model.modules()):
        if isinstance(layer, torch.nn.Linear):
            writer.add_histogram("Linear/weights-{}".format(linear), layer.weight,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            writer.add_histogram("Linear/bias-{}".format(linear), layer.bias,
                                 global_step=(epoch - 1) * len(train_loader) + batch_idx)
            linear += 1


def validation(config, writer, epoch, pretext_model, downstream_model, validation_loader,
               loss_function, format_logger, global_step, print_step):
    pretext_model.eval()
    downstream_model.eval()

    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, [data, target] in enumerate(validation_loader):
            # GPU 환경 설정
            if config['use_cuda']:
                data = data.cuda()
                target = target.cuda()
            loss, accuracy, z, c = pretext_model(data)

            _data = c[:, -1, :]
            output = downstream_model(_data)
            loss = loss_function(output, target)

            pred = output.max(1)[1].cpu()
            target = target.cpu()
            accuracy = accuracy_score(target, pred)

            # ...검증 중 손실(running loss)을 기록하고
            writer.add_scalar('Loss/spcls_valid_step', loss, global_step)
            writer.add_scalar('Accuracy/spcls_valid_step', accuracy, global_step)
            total_loss += loss
            total_accuracy += accuracy
            global_step += 1
            if batch_idx % print_step == 0:
                format_logger.info(
                    "[Epoch {}/{}] train step {:04d}/{:04d} \t"
                    "Loss = {:.4f} Accuracy = {:.4f}".format(
                        epoch, config['downstream_model']['epoch'],
                        batch_idx, len(validation_loader),
                        loss,
                        accuracy,
                    )
                )

        total_loss /= len(validation_loader)  # average loss
        total_accuracy /= len(validation_loader)  # average acc

        writer.add_scalar('Loss/spcls_valid_epoch', total_loss, epoch)
        writer.add_scalar('Accuracy/spcls_valid_epoch', total_accuracy, epoch)

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
