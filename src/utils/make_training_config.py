import json

name = 'direct_train10'

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
        "batch_size": 8,
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
        "train_dataset": "./dataset/train-list-librispeech.txt",
        "validation_dataset": "./dataset/dev-list-librispeech.txt",
        "num_workers": 8,
    },
    "model": {
        "epoch": 300,
        "batch_size": 256,
        "audio_window": 20480,
        "strides": [5, 4, 2, 2, 2],
        "filter_sizes": [10, 8, 4, 4, 4],
        "paddings": [2, 2, 2, 2, 1],
        "g_enc_hidden": 512,
        "g_ar_hidden": 256,
        "g_enc_input": 1,
    },
    "optimizer": {
        "optimizer_name": "Adam",
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "eps": 1e-09,
        "amsgrad": True,
        "betas": (0.9, 0.98),
    },
    "loss": {
      "learning_rate": 2.0e-4,
      "prediction_step": 12, # Time steps k to predict into future
      "negative_samples": 10, # Number of negative samples to be used for training
      "subsample": True, # Boolean to decide whether to subsample from the total sequence lengh within intermediate layers
      "calc_accuracy": True,
    },
    "tensorboard": {
        "writer_name": "runs/{}".format(name)
    },
    "checkpoint": {
        "save_directory_path": "./checkpoint",
        "file_name": "{}".format(name),
        "load_checkpoint": ''
    },
    "use_cuda": True,

}

if __name__ == '__main__':
    configuration_type = 2
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
