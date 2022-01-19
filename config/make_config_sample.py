import json

configuration = {
    # global
    "batch_size": 16,
    # dataset
    "train_dataset_filelist": "",
    "test_dataset_filelist": "",
    "validation_dataset_filelist": "",
    "speaker_filelist": None, # librispeech only or speaker classification task only
    "audio_window": 20480,
    "sample_rate": 16000,
    "augmentation": True,
    "full_audio": False,
    "use_librosa": False,
    "auto_trim": False,
    # dataloader
    "dataset_shuffle": True,
    "num_workers": 16,
}