import json

configuration = {
    "log_filename": "./log/training.log",
    "dataset": {
        "train_dataset": "./dataset/train-librispeech.h5",
        "train_id_set": "./dataset/train-librispeech.txt",
        "validation_dataset": "./dataset/dev-librispeech.h5",
        "validation_id_set": "./dataset/dev-librispeech.txt",
        "num_workers": 8,
    },
    "parameter": {
        "timestep": 12,
        "audio_window": 20840,
        "batch_size": 64,
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
        "epoch": 50,
    },
    "tensorboard": {
        "writer_name": "runs/test_space"
    },
    "checkpoint": {
        "save_directory_path": "./checkpoint",
        "file_name": "test01",
    },
    "use_cuda": False,
}

if __name__ == '__main__':
    filename = 'config_baseline.json'
    with open('../../config/{}'.format(filename), 'w', encoding='utf-8') as config_file:
        json.dump(configuration, config_file, indent='\t')
    print('successfully created')
