# pytorch library
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F

# custom library
import src.utils.interface_audio_io as audio_io
import src.utils.interface_audio_augmentation as audio_augmentation

# local library
import numpy as np
import random


def get_audio_file(file_list, index):
    audio_file = file_list[index]
    audio_file = audio_file[4:]  # audio file 위치에 따른 수정 코드
    return audio_file


def load_data_pipeline(audio_file, required_sample_rate, audio_window, full_audio, augmentation, cut_silence=None, custom_augmentation_list=None):
    waveform, sample_rate = audio_io.audio_loader("{}".format(audio_file))

    if cut_silence is not None:
        waveform = audio_io.cutoff(waveform, sample_rate, cut_silence[0], cut_silence[1])

    assert (
            sample_rate == required_sample_rate
    ), "sampling rate is not consistent throughout the dataset"
    waveform = audio_io.audio_adjust_length(waveform, audio_window)

    if not full_audio:
        waveform = audio_io.random_cutoff(waveform, audio_window)
    if augmentation:
        waveform = audio_augmentation.audio_augmentation_baseline(waveform, sample_rate, audio_window,
                                                                  custom_augmentation_list=custom_augmentation_list)
    if not full_audio:
        waveform = audio_io.audio_adjust_length(waveform, audio_window)
    return waveform


def load_data_pipeline_by_byol(audio_file, required_sample_rate, audio_window, full_audio, augmentation,
                               cut_silence=None):
    waveform, sample_rate = audio_io.audio_loader("{}".format(audio_file))

    if cut_silence is not None:
        waveform = audio_io.cutoff(waveform, sample_rate, cut_silence[0], cut_silence[1])

    assert (
            sample_rate == required_sample_rate
    ), "sampling rate is not consistent throughout the dataset"

    augmentation_list = [0, 2, 3, 4, 5, 6]
    # audio 길이 맞추기
    waveform = audio_io.audio_adjust_length(waveform, audio_window)
    pick_index = np.random.randint(waveform.shape[1] - audio_window + 1)
    if not full_audio:
        aug01_waveform = audio_io.random_cutoff(waveform, audio_window, pick_index)
        aug02_waveform = audio_io.random_cutoff(waveform, audio_window, pick_index)

    if augmentation:
        aug01_waveform = audio_augmentation.audio_augmentation_pipeline(aug01_waveform, sample_rate,
                                                                        audio_window,
                                                                        random.sample(augmentation_list, 3))
        aug02_waveform = audio_augmentation.audio_augmentation_pipeline(aug02_waveform, sample_rate,
                                                                        audio_window,
                                                                        random.sample(augmentation_list, 3))

    aug01_waveform = audio_io.audio_adjust_length(aug01_waveform, audio_window)
    aug02_waveform = audio_io.audio_adjust_length(aug02_waveform, audio_window)
    return aug01_waveform, aug02_waveform



# training CPC pretext model
class BaselineWaveformDataset(Dataset):
    def __init__(self, file_path: str, audio_window=20480, sample_rate=16000,
                 full_audio=False, augmentation=False):
        super(BaselineWaveformDataset, self).__init__()
        self.file_path = file_path
        self.audio_window = audio_window
        self.sample_rate = sample_rate
        self.full_audio = full_audio
        self.augmentation = augmentation

        # data file list
        id_data = open(self.file_path, 'r')
        self.file_list = [x.strip() for x in id_data.readlines()]
        id_data.close()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_file = get_audio_file(self.file_list, index)
        waveform = load_data_pipeline(audio_file, required_sample_rate=self.sample_rate,
                                      audio_window=self.audio_window, full_audio=self.full_audio,
                                      augmentation=self.augmentation)
        return waveform


# training waveBYOL base pretext model
class BaselineWaveformDatasetByBYOL(BaselineWaveformDataset):
    def __getitem__(self, index):
        audio_file = get_audio_file(self.file_list, index)
        aug01_waveform, aug02_waveform = load_data_pipeline_by_byol(audio_file, required_sample_rate=self.sample_rate,
                                                                    audio_window=self.audio_window,
                                                                    full_audio=self.full_audio,
                                                                    augmentation=self.augmentation)
        return aug01_waveform, aug02_waveform
