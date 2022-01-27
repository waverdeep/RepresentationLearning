import json

configuration = {
    # definition
    "use_cuda": True,
    "audio_window": 64000, # 20480 # 15200
    "sampling_rate": 16000,
    "epoch": 500,
    "batch_size": 18,
    "learning_rate": 0.0003,

    # dataset
    "dataset_type": "UrbanSound8KWaveformDataset",
    "dataset_name": "UrbanSound8K",
    "train_dataset": "./dataset/urbansound-train.txt",
    "test_dataset": "./dataset/urbansound-test.txt",
    "train_augmentation": True,
    "test_augmentation": False,
    "full_audio": False,

    # dataloader
    "num_workers": 16,
    "dataset_shuffle": True,
    "pin_memory": False,

    # model
    "pretext_model_name": "WaveBYOLEfficientB7",
    "pre_input_dims": 1,
    "pre_hidden_dims": 512,
    "pre_filter_sizes": [10, 8, 4, 4, 4],
    "pre_strides": [5, 4, 2, 2, 2],
    "pre_paddings": [2, 2, 2, 2, 1],
    "dimension": 491520, # 15200: 122880 # 20480: 163840
    "hidden_size": 512, # 512
    "projection_size": 4096, # 4096
    "ema_decay": 0.99,
    # optimizer
    "optimizer_name": "AdamP", # Adam # AdamP
    "weight_decay": 1e-2, # 0 # 1e-2
    "eps": 1e-08,
    "amsgrad": False,
    "betas": (0.9, 0.999), # (0.9, 0.999) # (0.9, 0.999)
    # checkpoint
    "checkpoint_save_directory_path": "./checkpoint",
}

if __name__ == '__main__':
    name = "pretext-{}-{}-{}".format(
        configuration['pretext_model_name'],
        configuration['dataset_name'],
        configuration['audio_window']
    )

    configuration["log_filename"] = "./log/{}".format(name)
    configuration["tensorboard_writer_name"] = "./runs/{}".format(name)
    configuration["checkpoint_file_name"] = "{}".format(name)

    filename = 'config-{}.json'.format(name)
    with open('./{}'.format(filename), 'w', encoding='utf-8') as config_file:
        json.dump(configuration, config_file, indent='\t')
