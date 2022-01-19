import json
name = "pretext-WAVEBYOLTest03-librispeech100-efficientb4aug-15200"


configuration = {
    # definition
    "log_filename": "./log/{}".format(name),
    "use_cuda": True,
    "audio_window": 15200,
    "sampling_rate": 16000,
    "epoch": 800,
    "batch_size": 16,
    "learning_rate": 0.0003,
    # dataset
    "dataset_type": "NormalWaveformDatasetByBYOL",
    "train_dataset": "./dataset/librispeech100-baseline-train.txt",
    "test_dataset": "./dataset/librispeech100-baseline-test.txt",
    "num_workers": 16,
    "dataset_shuffle": True,
    "pin_memory": True,
    "augmentation": True,
    "full_audio": False,
    "use_librosa": True,
    # model
    "pretext_model_name": "WaveBYOLTest03",
    "pre_input_dims": 1,
    "pre_hidden_dims": 512,
    "pre_filter_sizes": [10, 8, 4, 4, 4],
    "pre_strides": [5, 4, 2, 2, 2],
    "pre_paddings": [2, 2, 2, 2, 1],
    "dimension": 86016, #163840 eff b7, # 114688 eff b4, # 131072 resnet152
    "hidden_size": 512,
    "projection_size": 4096,
    "ema_decay": 0.99,
    # optimizer
    "optimizer_name": "Adam",
    "weight_decay": 0,
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
