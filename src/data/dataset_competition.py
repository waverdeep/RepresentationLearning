import torch
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
from collections import defaultdict
from src.utils import interface_file_io
torchaudio.set_audio_backend("sox_io")
import src.data.dataset_spectrogram as dataset_spectrogram
import torchaudio.transforms as T
import src.data.dataset_tool_speaker as speaker_tool
import src.utils.interface_audio_io as audio_io


def get_competition_speaker_list(file_list):
    speaker_list = []
    for index, file in enumerate(file_list):
        temp = file.split('/')
        speaker_id = temp[-2]
        speaker_list.append(speaker_id)
    return speaker_list


class CompetitionWaveformDataset(speaker_tool.NormalWaveformDataset):
    def __init__(self, directory_path, audio_window=20480):
        self.directory_path = directory_path
        self.audio_window = audio_window

        self.file_list = []
        id_data = open(self.directory_path, 'r')
        # strip() 함수를 사용해서 뒤에 개행을 제거
        self.file_list = [x.strip() for x in id_data.readlines()]
        id_data.close()

        self.speaker_list_file = open("./dataset/speaker_recognition-train-20480.txt", 'r')
        self.speaker_file_list = [x.strip() for x in self.speaker_list_file.readlines()]
        self.speaker_list = get_competition_speaker_list(self.speaker_file_list)
        self.speaker_dict = speaker_tool.get_speaker_dict(self.speaker_list)

        self.vad = T.Vad(16000)

    # speaker_recognition/dev/0001/wav.wav
    def __getitem__(self, index):
        audio_file = self.file_list[index]
        temp = audio_file.split('/')
        speaker_id = temp[-2]
        audio_file = audio_file[4:]
        waveform, sampling_rate = audio_io.audio_loader("{}".format(audio_file))
        assert (
                sampling_rate == 16000
        ), "sampling rate is not consistent throughout the dataset"

        waveform = self.vad(waveform)
        waveform = torch.flip(waveform, [0, 1])
        waveform = self.vad(waveform)
        waveform = torch.flip(waveform, [0, 1])

        while True:
            audio_length = waveform.shape[1]
            if audio_length < self.audio_window:
                waveform = torch.cat((waveform, waveform), 1)
            else:
                break

        audio_length = waveform.shape[1]
        random_index = np.random.randint(audio_length - self.audio_window + 1)
        waveform = waveform[:, random_index: random_index + self.audio_window]
        return waveform, 0, speaker_id
