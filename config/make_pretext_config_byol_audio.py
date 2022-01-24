import json
name = "pretext-BYOLA-urbansound-training01-batch256"


configuration = {
    # definition
    "log_filename": "./log/{}".format(name),
    "use_cuda": True,
    "audio_window": 15200,
    "sampling_rate": 16000,
    "epoch": 500,
    "batch_size": 512,
    "learning_rate": 0.0003,
    # dataset
    "dataset_type": "ByolAudioDataset",
    "train_dataset": "./dataset/urbansound-train.txt",
    "test_dataset": "./dataset/urbansound-test.txt",
    "num_workers": 16,
    "dataset_shuffle": True,
    "pin_memory": True,
    "augmentation": False,
    "full_audio": False,
    "use_librosa": True,
    # model
    "pretext_model_name": "BYOLAudioModel",
    "input_dims": [1, 64, 64],
    "hidden_dims": [64, 64, 64],
    "strides": [1, 1, 1],
    "filter_sizes": [3, 3, 3],
    "paddings": [1, 1, 1],
    "maxpool_filter_sizes": [2, 2, 2],
    "maxpool_strides": [2, 2, 2],
    "feature_dimension": 2048,
    "hidden_size": 256,
    "projection_size": 4096,
    "ema_decay": 0.99,
    # feature
    "sample_rate": 16000,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 160,
    "n_mels": 64,
    "f_min": 60,
    "f_max": 7800,
    "shape": [64, 96],
    # optimizer
    "optimizer_name": "Adam",
    "weight_decay": 0.99,
    "eps": 1e-08,
    "amsgrad": False,
    "betas": (0.9, 0.999),
    # tensorboard
    "tensorboard_writer_name": "./runs/{}".format(name),
    # checkpoint
    "checkpoint_save_directory_path": "./checkpoint",
    "checkpoint_file_name": "{}".format(name),
}


if __name__ == '__main__':
    filename = 'config_{}.json'.format(name)
    with open('../../config/{}'.format(filename), 'w', encoding='utf-8') as config_file:
        json.dump(configuration, config_file, indent='\t')
