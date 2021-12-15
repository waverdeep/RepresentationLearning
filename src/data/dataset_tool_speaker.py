from collections import defaultdict
import src.utils.interface_audio_io as audio_io
import torch
import numpy as np
from torch.utils.data import Dataset


def get_librispeech_speaker_list(file_list):
    speaker_list = []
    for index, file in enumerate(file_list):
        filename = file.split("/")[-1]
        filename = filename.split(".")[0]
        speaker_id = filename.split("-")[0] # speaker_id, dir_id, sample_id
        speaker_list.append(speaker_id)
    return speaker_list


def get_speaker_dict(speaker_list):
    speaker_id_dict = {}
    for idx, key in enumerate(sorted(list(set(speaker_list)))):
        speaker_id_dict[key] = idx
    return speaker_id_dict


def get_speaker_align(file_list):
    item_list = []
    speaker_align = defaultdict(list)
    index = 0
    for index, file in enumerate(file_list):
        filename = file.split("/")[-1]
        filename = filename.split(".")[0]
        speaker_id, dir_id, sample_id = filename.split("-")
        item_list.append((speaker_id, dir_id, sample_id))
        speaker_align[speaker_id].append(index)
        index += 1

    return item_list, speaker_align


def audio_auto_trim(waveform, vad, audio_window=None):
    waveform = vad(waveform)
    waveform = torch.flip(waveform, [0, 1])
    waveform = vad(waveform)
    waveform = torch.flip(waveform, [0, 1])

    if audio_window is not None:
        while True:
            audio_length = waveform.shape[1]
            if audio_length < audio_window:
                waveform = torch.cat((waveform, waveform), 1)
            else:
                break
    return waveform


def random_cutoff(waveform, audio_window):
    audio_length = waveform.shape[1]
    random_index = np.random.randint(audio_length - audio_window + 1)
    cutoff_waveform = waveform[:, random_index: random_index + audio_window]
    return cutoff_waveform


class NormalWaveformDataset(Dataset):
    def __init__(self, directory_path, audio_window=20480, full_audio=False):
        self.directory_path = directory_path
        self.audio_window = audio_window
        self.full_audio = full_audio

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
        if not self.full_audio:
            waveform = random_cutoff(waveform, self.audio_window)
        return waveform, 0, 0