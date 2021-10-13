import torch
import argparse
import json
import src.data.dataset as dataset
import src.utils.logger as logger
import src.models.model_baseline as model_baseline
import src.optimizers.optimizer_baseline as optimizer_baseline


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
    format_logger.info('configurations: {}'.format(config))
    # 학습 함수 시작
    train(config=config, format_logger=format_logger)


def train(config, format_logger):
    format_logger.info("load train/validation dataset ...")
    train_loader = dataset.get_dataloader(dataset=config['dataset']['train_dataset'],
                                          id_set=config['dataset']['train_id_set'],
                                          audio_window=config['parameter']['audio_window'],
                                          batch_size=config['parameter']['batch_size'],
                                          num_workers=['dataset']['num_workers'],
                                          shuffle=True, pin_memory=False)
    validation_loader = dataset.get_dataloader(dataset=config['dataset']['validation_dataset'],
                                               id_set=config['dataset']['validation_id_set'],
                                               audio_window=config['parameter']['audio_window'],
                                               batch_size=config['parameter']['batch_size'],
                                               num_workers=['dataset']['num_workers'],
                                               shuffle=False, pin_memory=False)

    format_logger.info("load model ...")
    model = model_baseline.CPC(timestep=config['parameter']['timestep'], audio_window=config['parameter']['audio_window'])
    if config['use_cuda']:
        model = model.cuda()
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    format_logger.info(">>> model_structure <<<")
    format_logger.info("{}".format(model))
    format_logger.info("model parameters: {}".format(model_params))

    optimizer = optimizer_baseline.get_optimizer(name=args.optimizer, model_parameter=model.parameters(), learning_rate=args.learning_rate, weight_decay=args.weight_decay, optimizer_param={})







if __name__ == '__main__':
    main()
