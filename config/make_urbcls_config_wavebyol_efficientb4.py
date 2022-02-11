import json

configuration = {
    # definition
    "use_cuda": True,
    "audio_window": 64000, # 20480 # 15200
    "sampling_rate": 16000,
    "epoch": 200,
    "batch_size": 128,
    "learning_rate": 0.001,

    # dataset
    "dataset_type": "UrbanSound8KWaveformDataset",
    "dataset_name": "UrbanSound8K",
    "train_dataset": "./dataset/urbansound-train.txt",
    "test_dataset": "./dataset/urbansound-test.txt",
    "train_augmentation": True,
    "test_augmentation": False,
    "full_audio": False,
    "metadata": "./dataset/UrbanSound8K/metadata/UrbanSound8K.csv",

    # dataloader
    "num_workers": 8,
    "dataset_shuffle": True,
    "pin_memory": False,

    # model
    "pretext_model_name": "WaveBYOLEfficientB4",
    "pre_input_dims": 1,
    "pre_hidden_dims": 512,
    "pre_filter_sizes": [10, 8, 4, 4, 4],
    "pre_strides": [5, 4, 2, 2, 2],
    "pre_paddings": [2, 2, 2, 2, 1],
    "dimension": 64, # 15200: 86016 # 20480: 1146884096
    "hidden_size": 2048, # 512
    "projection_size": 4096,
    "ema_decay": 0.99,
    # downstream model
    "downstream_model_name": "NormalClassification",
    "downstream_input_dim": 4096,
    "downstream_hidden_dim": 4096,
    "downstream_output_dim": 10,
    # optimizer
    "optimizer_name": "AdamP",
    "weight_decay": 0,
    "eps": 1e-08,
    "amsgrad": False,
    "betas": (0.9, 0.999),

    # loss function
    "loss_function": "CrossEntropyLoss",

    # checkpoint
    "checkpoint_save_directory_path": "./checkpoint",
    "pretext_checkpoint": './checkpoint/pretext-WaveBYOLEfficientB4-FSD50K-20480-2048/pretext-WaveBYOLEfficientB4-FSD50K-20480-2048-model-best-2022_2_2_7_47_40-epoch-306.pt',
    "downstream_checkpoint": None,
}


if __name__ == '__main__':
    name = "downstream-{}-{}-{}-{}".format(
        configuration['pretext_model_name'],
        configuration['dataset_name'],
        configuration['audio_window'],
        configuration['hidden_size']
    )

    configuration["log_filename"] = "./log/{}".format(name)
    configuration["tensorboard_writer_name"] = "./runs/{}".format(name)
    configuration["checkpoint_file_name"] = "{}".format(name)

    filename = 'config-{}.json'.format(name)
    with open('./{}'.format(filename), 'w', encoding='utf-8') as config_file:
        json.dump(configuration, config_file, indent='\t')
