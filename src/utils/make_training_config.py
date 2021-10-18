import json

name = 'direct_train02'

configuration_type01 = {
    "log_filename": "./log/{}.log".format(name),
    "dataset": {
        "train_dataset": "./dataset/train-list-librispeech.txt",
        # "train_id_set": "./dataset/train-librispeech.txt",
        "validation_dataset": "./dataset/dev-list-librispeech.txt",
        # "validation_id_set": "./dataset/dev-librispeech.txt",
        "num_workers": 8,
    },
    "parameter": {
        "timestep": 12,
        "audio_window": 20480,
        "batch_size": 512,
    },
    "optimizer": {
        "optimizer_name": "Adam",
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "eps": 1e-09,
        "amsgrad": True,
        "betas": (0.9, 0.98),
    },
    "train": {
        "epoch": 3000,
    },
    "tensorboard": {
        "writer_name": "runs/{}".format(name)
    },
    "checkpoint": {
        "save_directory_path": "./checkpoint",
        "file_name": "{}".format(name),
    },
    "use_cuda": True,
}
configuration = {
    "log_filename": "./log/{}.log".format(name),
    "dataset": {
        "train_dataset": './dataset/train-list-librispeech.txt',
        "validation_dataset": './dataset/dev-list-librispeech.txt',
        "num_workers": 8,
    },
    "parameter": {
        "timestep": 12,
        "audio_window": 20480,
        "batch_size": 512,
    },
    "optimizer": {
        "optimizer_name": "Adam",
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "eps": 1e-09,
        "amsgrad": True,
        "betas": (0.9, 0.98),
    },
    "train": {
        "epoch": 3000,
    },
    "tensorboard": {
        "writer_name": "runs/{}".format(name)
    },
    "checkpoint": {
        "save_directory_path": "./checkpoint",
        "file_name": "{}".format(name),
    },
    "use_cuda": True,
}

configuration_type02 = {
    "log_filename": "./log/{}.log".format(name),
    "dataset": {
        "train_dataset": ".train-list-librispeech.txt",
        # "train_id_set": "./dataset/train-librispeech.txt",
        "validation_dataset": "dev-list-librispeech.txt",
        # "validation_id_set": "./dataset/dev-librispeech.txt",
        "num_workers": 8,
    },
    "tensorboard": {
        "writer_name": "runs/{}".format(name)
    },
    "checkpoint": {
        "save_directory_path": "./checkpoint",
        "file_name": "{}".format(name),
    },
    "use_cuda": True,

}

if __name__ == '__main__':
    configuration_type = 1
    if configuration_type == 1:
        filename = 'config_type01_{}.json'.format(name)
        with open('../../config/{}'.format(filename), 'w', encoding='utf-8') as config_file:
            json.dump(configuration_type01, config_file, indent='\t')
        print('successfully created')
    elif configuration_type == 2:
        filename = 'config_type02_{}.json'.format(name)
        with open('../../config/{}'.format(filename), 'w', encoding='utf-8') as config_file:
            json.dump(configuration_type02, config_file, indent='\t')
        print('successfully created')
