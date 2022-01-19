import json
name = "spkcls-WAVEBYOLTest03-librispeech100-efficientb4aug-batch64"


configuration = {
    # definition
    "log_filename": "./log/{}".format(name),
    "use_cuda": True,
    "audio_window": 20480,
    "sampling_rate": 16000,
    "epoch": 800,
    "batch_size": 256,
    "learning_rate": 0.0003,
    # dataset
    "dataset_type": "LibriSpeechWaveformDataset",
    "train_dataset": "./dataset/librispeech100-baseline-train.txt",
    "test_dataset": "./dataset/librispeech100-baseline-test.txt",
    "num_workers": 16,
    "dataset_shuffle": True,
    "pin_memory": True,
    "augmentation": False,
    "full_audio": False,
    "use_librosa": True,
    # pretext model
    "pretext_model_name": "WaveBYOLTest03",
    "pre_input_dims": 1,
    "pre_hidden_dims": 512,
    "pre_filter_sizes": [10, 8, 4, 4, 4],
    "pre_strides": [5, 4, 2, 2, 2],
    "pre_paddings": [2, 2, 2, 2, 1],
    "dimension": 114688,
    "hidden_size": 512,
    "projection_size": 4096,
    "ema_decay": 0.99,
    # downstream model
    "downstream_model_name": "NormalClassification",
    "downstream_input_dim": 114688,
    "downstream_hidden_dim": 4096,
    "downstream_output_dim": 251,
    # loss
    "loss_function": "CrossEntropyLoss",
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
    "pretext_checkpoint": './checkpoint/pretext-WAVEBYOLTest03-librispeech100-efficientb4aug-batch64-model-best-2022_1_17_7_10_20-epoch-207.pt',
    "downstream_checkpoint": None,
}


if __name__ == '__main__':
    filename = 'config_{}.json'.format(name)
    with open('../../config/{}'.format(filename), 'w', encoding='utf-8') as config_file:
        json.dump(configuration, config_file, indent='\t')
