import torch
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import src.data.dataset as dataset
import src.utils.logger as logger
import src.models.model_baseline as model_baseline
import src.optimizers.optimizer_baseline as optimizer_baseline
import src.utils.setup_tensorboard as tensorboard


def main():
    # configruation 파라미터를 argparse로 받아오기
    parser = argparse.ArgumentParser(description='pytorch representation learning')
    parser.add_argument('--configuration', required=False, default='./config/config_baseline.json')
    args = parser.parse_args()

    # 학습에 필요한 모든 configuration 및 파라미터들을 json으로 불러오기
    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)
    # 로거 생성하기
    format_logger = logger.setup_log(save_filename=config['log_filename'])
    # 모델 configuration 출력
    format_logger.info('configurations: {}'.format(config))

    format_logger.info("load train/validation dataset ...")
    # 학습 데이터로더 생성
    train_loader = dataset.get_dataloader(dataset=config['dataset']['train_dataset'],
                                          id_set=config['dataset']['train_id_set'],
                                          audio_window=config['parameter']['audio_window'],
                                          batch_size=config['parameter']['batch_size'],
                                          num_workers=config['dataset']['num_workers'],
                                          shuffle=True, pin_memory=False)
    # 검증 데이터로더 생성
    validation_loader = dataset.get_dataloader(dataset=config['dataset']['validation_dataset'],
                                               id_set=config['dataset']['validation_id_set'],
                                               audio_window=config['parameter']['audio_window'],
                                               batch_size=config['parameter']['batch_size'],
                                               num_workers=config['dataset']['num_workers'],
                                               shuffle=False, pin_memory=False)


    format_logger.info("load model ...")
    # 모델 생성
    model = model_baseline.CPC(timestep=config['parameter']['timestep'],
                               audio_window=config['parameter']['audio_window'])

    # 텐서보드 셋업
    writer = tensorboard.set_tensorboard_writer(config['tensorboard']['writer_name'])

    # # 텐서보드로 모델 구조 보여주기
    # tensorboard.show_model_tensorboard_with_no_label(writer, model, train_loader, config['parameter']['batch_size'])

    # GPU 환경 설정
    if config['use_cuda']:
        model = model.cuda()

    # 파라미터 개수 측정
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 모델 정보 출력
    format_logger.info(">>> model_structure <<<")
    format_logger.info("model parameters: {}".format(model_params))
    format_logger.info("{}".format(model))

    # 옵티마이저 생성
    optimizer = optimizer_baseline.get_optimizer(model_parameter=model.parameters(),
                                                 optimizer_param=config['optimizer'])

    # 학습 코드 작성
    best_accuracy = 0
    # loss의 기본값을 무한대로 세팅
    best_loss = np.inf
    best_epoch = 0

    num_of_epoch = config['train']['epoch']
    for epoch in range(num_of_epoch):
        epoch = epoch + 1

        # 학습 함수 시작
        format_logger.info("start training ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        train(config=config,
              writer=writer,
              epoch=epoch,
              model=model,
              train_loader=train_loader,
              optimizer=optimizer,
              format_logger=format_logger)

        # 검증 함수 시작
        format_logger.info("start validating ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        validation_accuracy, validation_loss = validation(config=config,
                                                          writer=writer,
                                                          epoch=epoch,
                                                          model=model,
                                                          validation_dataloader=validation_loader,
                                                          format_logger=format_logger)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            save_checkpoint(config=config,
                            model=model,
                            optimizer=optimizer,
                            loss=validation_loss,
                            epoch=best_epoch,
                            format_logger=format_logger)



    # 텐서보드 종료
    tensorboard.close_tensorboard_writer(writer)


def train(config, writer, epoch, model, train_loader, optimizer, format_logger):
    train_bar = tqdm(train_loader,
                     desc='{}/{} epoch training ... '.format(epoch, config['train']['epoch']))
    for batch_idx, data in enumerate(train_bar):
        # convolution 연산을 위채 1채널 추가
        data = data.float().unsqueeze(1)
        # GPU 환경 설정
        if config['use_cuda']:
            data = data.cuda()

        # 옵티마이저 초기화
        optimizer.zero_grad()
        # gru 모델에 들어간 hidden state 설정하기
        hidden = model.init_hidden(len(data), config['use_cuda'])
        accuracy, loss, hidden = model(data, hidden)
        train_bar.set_description('{}/{} epoch [ acc: {}/ loss: {} ]'.format(
            epoch, config['train']['epoch'], round(float(accuracy), 3), round(float(loss), 3)))
        loss.backward()
        optimizer.step()

        # ...학습 중 손실(running loss)을 기록하고
        writer.add_scalar('training loss', loss, (epoch-1) * len(train_loader) + batch_idx)
        writer.add_scalar('training accuracy', accuracy * 100, (epoch-1) * len(train_loader) + batch_idx)


def validation(config, writer, epoch, model, validation_dataloader, format_logger):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        validation_bar = tqdm(validation_dataloader,
                              desc='{}/{} epoch validating ... '.format(epoch, config['train']['epoch']))
        for batch_idx, data in enumerate(validation_bar):
            # convolution 연산을 위채 1채널 추가
            data = data.float().unsqueeze(1)
            # GPU 환경 설정
            if config['use_cuda']:
                data = data.cuda()

            hidden = model.init_hidden(len(data), config['use_cuda'])
            accuracy, loss, hidden = model(data, hidden)

            validation_bar.set_description('{}/{} epoch [ acc: {}/ loss: {} ]'.format(
                                           epoch, config['train']['epoch'], round(float(accuracy), 3), round(float(loss), 3)))

            # ...검증 중 손실(running loss)을 기록하고
            writer.add_scalar('validating loss', loss, (epoch - 1) * len(validation_dataloader) + batch_idx)
            writer.add_scalar('validating accuracy', accuracy * 100, (epoch - 1) * len(validation_dataloader) + batch_idx)

            total_loss += len(data) * loss
            total_accuracy += len(data) * accuracy

        total_loss /= len(validation_dataloader.dataset)  # average loss
        total_accuracy /= len(validation_dataloader.dataset)  # average acc

        format_logger.info("[ {}/{} epoch validation result: [ average acc: {}/ average loss: {} ]".format(
            epoch, config['train']['epoch'], total_accuracy, total_loss
        ))

    return total_accuracy, total_loss


def save_checkpoint(config, model, optimizer, loss, epoch,format_logger):
    file_path = os.path.join(config['checkpoint']['save_direcroy_path'],
                             config['checkpoint']['file_name']+"-model_best.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "loss": loss}, file_path)
    format_logger.info("saved best checkpoint ... ")


if __name__ == '__main__':
    main()