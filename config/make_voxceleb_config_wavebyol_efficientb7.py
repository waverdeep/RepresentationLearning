import json

configuration = {
    # definition
    "use_cuda": True,
    "audio_window": 96000, # 20480 # 15200
    "sampling_rate": 16000,
    "epoch": 200,
    "batch_size": 64,
    "learning_rate": 0.0003,

    # dataset
    "dataset_type": "VoxCelebWaveformDataset",
    "dataset_name": "VoxCeleb",
    "train_dataset": "./dataset/voxceleb01-train.txt",
    "test_dataset": "./dataset/voxceleb01-test.txt",
    "train_augmentation": True,
    "test_augmentation": False,
    "full_audio": False,

    # dataloader
    "num_workers": 8,
    "dataset_shuffle": True,
    "pin_memory": False,

    ## model
    "pretext_model_name": "WaveBYOLEfficientB7",
    "pre_input_dims": 1,
    "pre_hidden_dims": 512,
    "pre_filter_sizes": [10, 8, 4, 4, 4],
    "pre_strides": [5, 4, 2, 2, 2],
    "pre_paddings": [2, 2, 2, 2, 1],
    "dimension": 64, # 15200: 122880 # 20480: 163840
    "hidden_size": 512, # 512
    "projection_size": 4096, # 4096
    "ema_decay": 0.99,
    # optimizer
    "optimizer_name": "AdamP",
    # downstream model
    "downstream_model_name": "NormalClassification",
    "downstream_input_dim": 4096,
    "downstream_hidden_dim": 4096,
    "downstream_output_dim": 40,

    # loss function
    "loss_function": "CrossEntropyLoss",

    # checkpoint
    "checkpoint_save_directory_path": "./checkpoint",
    "pretext_checkpoint": './checkpoint/pretext-WaveBYOLEfficientB7-FSD50K-20480-model-best-2022_2_2_13_54_6-epoch-242.pt',
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
