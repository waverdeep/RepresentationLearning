import json
name = "pretext-GCPC-librispeech100-training01-batch256"


configuration = {
    # definition
    "log_filename": "./log/{}".format(name),
    "use_cuda": True,
    "audio_window": 20480,
    "epoch": 800,
    "batch_size": 256,
    "learning_rate": 2.0e-4,
    # dataset
    "dataset_type": "LibriSpeechWaveformDataset",
    "train_dataset": "./dataset/librispeech100-baseline-train.txt",
    "test_dataset": "./dataset/librispeech100-baseline-test.txt",
    "num_workers": 24,
    "dataset_shuffle": True,
    "pin_memory": True,
    "augmentation": False,
    # model
    "pretext_model_name": "GenerativeCPCModel",
    "strides": [5, 4, 2, 2, 2],
    "filter_sizes": [10, 8, 4, 4, 4],
    "paddings": [2, 2, 2, 2, 1],
    "g_enc_hidden": 512,
    "g_ar_hidden": 256,
    "g_enc_input": 1,
    # optimizer
    "optimizer_name": "Adam",
    "weight_decay": 0,
    "eps": 1e-08,
    "amsgrad": False,
    "betas": (0.9, 0.999),
    # loss
    "prediction_step": 12, # Time steps k to predict into future
    "negative_samples": 10, # Number of negative samples to be used for training
    "subsample": True, # Boolean to decide whether to subsample from the total sequence lengh within intermediate layers
    "calc_accuracy": True,
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
