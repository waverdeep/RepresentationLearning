from torch.utils.data import Dataset
import src.utils.interface_audio_io as audio_io
import src.utils.interface_audio_augmentation as audio_augmentation
import numpy as np
import torch
import random


class NormalWaveformDataset(Dataset):
    def __init__(self, directory_path, audio_window=20480, full_audio=False, augmentation=False):
        self.directory_path = directory_path
        self.audio_window = audio_window
        self.full_audio = full_audio
        self.augmentation = augmentation

        self.file_list = []
        id_data = open(self.directory_path, 'r')
        # strip() 함수를 사용해서 뒤에 개행을 제거
        self.file_list = [x.strip() for x in id_data.readlines()]
        id_data.close()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_file = self.file_list[index]
        audio_file = audio_file[4:]
        waveform, sampling_rate = audio_io.audio_loader("{}".format(audio_file))
        # sampling rate가 16000가 아니면 에러 메시지를 띄워줄 수 있도록 함
        assert (
                sampling_rate == 16000
        ), "sampling rate is not consistent throughout the dataset"

        if self.augmentation:
            waveform = audio_augmentation.audio_augment_baseline(waveform, sampling_rate)

        if not self.full_audio:
            waveform = audio_io.random_cutoff(waveform, self.audio_window)
        return waveform, 0, 0


class NormalWaveformDatasetByBYOL(NormalWaveformDataset):
    def __getitem__(self, index):
        audio_file = self.file_list[index]
        audio_file = audio_file[4:]
        waveform, sampling_rate = audio_io.audio_loader("{}".format(audio_file))
        augmentation_list = [0, 1, 2, 3, 4, 5]

        # sampling rate가 16000가 아니면 에러 메시지를 띄워줄 수 있도록 함
        assert (
                sampling_rate == self.sampling_rate
        ), "sampling rate is not consistent throughout the dataset"

        waveform = audio_io.audio_adjust_length(waveform, self.audio_window)
        pick_index = np.random.randint(waveform.shape[1] - self.audio_window + 1)
        if not self.full_audio:
            aug01_waveform = audio_io.random_cutoff(waveform, self.audio_window, pick_index)
            aug02_waveform = audio_io.random_cutoff(waveform, self.audio_window, pick_index)

        if self.augmentation:
            aug01_waveform = audio_augmentation.audio_augmentation_pipeline(aug01_waveform, sampling_rate,
                                                                            self.audio_window,
                                                                            random.sample(augmentation_list, 3))
            aug02_waveform = audio_augmentation.audio_augmentation_pipeline(aug02_waveform, sampling_rate,
                                                                            self.audio_window,
                                                                            random.sample(augmentation_list, 3))

        aug01_waveform = audio_io.audio_adjust_length(aug01_waveform, self.audio_window)
        aug02_waveform = audio_io.audio_adjust_length(aug02_waveform, self.audio_window)
        return aug01_waveform, aug02_waveform, 0, 0