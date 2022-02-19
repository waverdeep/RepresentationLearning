import json

configuration = {
    # definition
    "use_cuda": True,
    "audio_window": 15200,
    "sampling_rate": 16000,
    "epoch": 200,
    "batch_size": 32,
    "learning_rate": 0.001,

    # dataset
    "dataset_type": "UrbanSound8KWaveformDataset",
    "dataset_name": "UrbanSound8K",
    "train_dataset": "./dataset/urbansound-train.txt",
    "test_dataset": "./dataset/urbansound-test.txt",
    "train_augmentation": True,
    "test_augmentation": False,
    "full_audio": False,

    # dataloader
    "num_workers": 8,
    "dataset_shuffle": True,
    "pin_memory": False,

    ## model
    "pretext_model_name": "FeatBYOLLightEfficientB0",
    "encoder_input_dim": 1,
    "encoder_hidden_dim": 512,
    "encoder_filter_size": [10, 8, 4, 4],
    "encoder_stride": [5, 4, 2, 2],
    "encoder_padding": [2, 2, 2, 1],
    "efficientnet_version": "b0",
    "mlp_input_dim": 1280,
    "mlp_hidden_dim": 4096,
    "mlp_output_dim": 4096,
    "ema_decay": 0.99,
    # optimizer
    "optimizer_name": "Adam",
    # downstream model
    "downstream_model_name": "DownstreamClassificationWithReshape",
    "downstream_input_dim": 1280,
    "downstream_hidden_dim": 1024,
    "downstream_output_dim": 10,

    # loss function
    "loss_function": "CrossEntropyLoss",

    # checkpoint
    "checkpoint_save_directory_path": "./checkpoint",
    "pretext_checkpoint": './checkpoint/'
                          'pretext-FeatBYOLLightEfficientB0-FSD50K-15200/'
                          'pretext-FeatBYOLLightEfficientB0-FSD50K-15200-model-best-2022_2_18_11_35_9-epoch-42.pt',
    "downstream_checkpoint": None,
}


if __name__ == '__main__':
    name = "downstream-{}-{}-{}".format(
        configuration['pretext_model_name'],
        configuration['dataset_name'],
        configuration['audio_window'],
    )

    configuration["log_filename"] = "./log/{}".format(name)
    configuration["tensorboard_writer_name"] = "./runs/{}".format(name)
    configuration["checkpoint_file_name"] = "{}".format(name)

    filename = 'config-{}.json'.format(name)
    with open('./{}'.format(filename), 'w', encoding='utf-8') as config_file:
        json.dump(configuration, config_file, indent='\t')
