import torch
import torchaudio
from torch.utils.data import Dataset
import random
import numpy as np
import torchaudio.transforms as T
import src.utils.interface_audio_io as audio_io
import src.data.dataset_tool_speaker as speaker_tool
import src.utils.interface_audio_augmentation as audio_augmentation
torchaudio.set_audio_backend("sox_io")
import torch.nn.functional as F


class LibriSpeechWaveformDataset(Dataset):
    def __init__(self, directory_path, audio_window=20480, sampling_rate=16000, auto_trim=False, full_audio=False,
                 augmentation=False):
        self.directory_path = directory_path
        self.audio_window = audio_window
        self.sampling_rate = sampling_rate
        self.auto_trim = auto_trim
        self.full_audio = full_audio
        self.augmentation = augmentation

        self.file_list = []
        id_data = open(self.directory_path, 'r')
        # strip() 함수를 사용해서 뒤에 개행을 제거
        self.file_list = [x.strip() for x in id_data.readlines()]
        id_data.close()

        self.speaker_list = speaker_tool.get_librispeech_speaker_list(self.file_list)
        self.speaker_dict = speaker_tool.get_speaker_dict(self.speaker_list)
        _, self.speaker_align = speaker_tool.get_speaker_align(self.file_list)

        if self.auto_trim:
            self.vad = T.Vad(sampling_rate)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_file = self.file_list[index]
        audio_file = audio_file[4:]
        waveform, sampling_rate = audio_io.audio_loader("{}".format(audio_file))
        filename = audio_file.split("/")[-1]
        filename = filename.split(".")[0] # speaker_id, dir_id, sample_id
        speaker_id = filename.split("-")[0]

        # sampling rate가 16000가 아니면 에러 메시지를 띄워줄 수 있도록 함
        assert (
            sampling_rate == self.sampling_rate
        ), "sampling rate is not consistent throughout the dataset"

        if self.auto_trim:
            waveform = audio_io.audio_auto_trim(waveform, self.vad, self.audio_window)

        if self.augmentation:
            waveform = audio_augmentation.audio_augment_baseline(waveform, sampling_rate)

        if not self.full_audio:
            waveform = audio_io.random_cutoff(waveform, self.audio_window)

        return waveform, str(filename), str(speaker_id)

    def get_audio_by_speaker(self, speaker_id, batch_size):
        batch_size = min(len(self.speaker_align[speaker_id]), batch_size)
        batch = torch.zeros(batch_size, 1, self.audio_window)
        for index in range(batch_size):
            batch[index, 0, :], _, _ = self.__getitem__(self.speaker_align[speaker_id][index])
        return batch


class LibriSpeechWaveformDatasetByBYOL(LibriSpeechWaveformDataset):
    def __getitem__(self, index):
        # 오디오파일 가져오기
        audio_file = self.file_list[index]
        audio_file = audio_file[4:]
        # 오디오파일 읽기 (loader)
        waveform, sampling_rate = audio_io.audio_loader("{}".format(audio_file))
        # speaker_id, sample_id 추출
        filename = audio_file.split("/")[-1]
        filename = filename.split(".")[0]  # speaker_id, dir_id, sample_id
        speaker_id = filename.split("-")[0]

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
            aug01_waveform = audio_augmentation.audio_augmentation_pipeline(aug01_waveform, sampling_rate, self.audio_window,
                                                                            random.sample(augmentation_list, 3))
            aug02_waveform = audio_augmentation.audio_augmentation_pipeline(aug02_waveform, sampling_rate, self.audio_window,
                                                                            random.sample(augmentation_list, 3))

        aug01_waveform = audio_io.audio_adjust_length(aug01_waveform, self.audio_window)
        aug02_waveform = audio_io.audio_adjust_length(aug02_waveform, self.audio_window)


        return aug01_waveform, aug02_waveform, str(filename), str(speaker_id)