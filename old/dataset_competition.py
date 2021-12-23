import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
torchaudio.set_audio_backend("sox_io")
import torchaudio.transforms as T
import src.data.dataset_tool_speaker as speaker_tool
import src.utils.interface_audio_io as audio_io
import src.data.dataset_normal as normal
import warnings
warnings.filterwarnings(action='ignore')


class CompetitionWaveformDataset(normal.NormalWaveformDataset):
    def __init__(self, directory_path, audio_window=20480, full_audio=False):
        self.directory_path = directory_path
        self.audio_window = audio_window
        self.full_audio = full_audio
        self.file_list = []
        id_data = open(self.directory_path, 'r')
        # strip() 함수를 사용해서 뒤에 개행을 제거
        self.file_list = [x.strip() for x in id_data.readlines()]
        id_data.close()

        self.speaker_list_file = open("./dataset/speaker_recognition-train-20480.txt", 'r')
        self.speaker_file_list = [x.strip() for x in self.speaker_list_file.readlines()]
        self.speaker_list = speaker_tool.get_competition_speaker_list(self.speaker_file_list)
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

        waveform = audio_io.audio_auto_trim(waveform, vad=self.vad, audio_window=self.audio_window)

        if not self.full_audio:
            waveform = audio_io.random_cutoff(waveform, audio_window=self.audio_window)

        return waveform, 0, speaker_id


class CompetitionMFCCDataset(Dataset):
    def __init__(self, directory_path, audio_window=20480, sampling_size=16000,
                 n_fft=2048, win_length=1024, hop_length=512, n_mels=256, n_mfcc=256):
        self.directory_path = directory_path
        self.audio_window = audio_window
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        self.file_list = []
        id_data = open(self.directory_path, 'r')
        # strip() 함수를 사용해서 뒤에 개행을 제거
        self.file_list = [x.strip() for x in id_data.readlines()]
        id_data.close()

        self.speaker_list_file = open("./dataset/speaker_recognition-train-20480.txt", 'r')
        self.speaker_file_list = [x.strip() for x in self.speaker_list_file.readlines()]
        self.speaker_list = speaker_tool.get_competition_speaker_list(self.speaker_file_list)
        self.speaker_dict = speaker_tool.get_speaker_dict(self.speaker_list)

        self.vad = T.Vad(sampling_size)
        self.mfcc_transform = T.MFCC(
            sample_rate=sampling_size,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "n_mels": self.n_mels,
                "hop_length": self.hop_length,
            }
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_file = self.file_list[index]
        temp = audio_file.split('/')
        speaker_id = temp[-2]
        audio_file = audio_file[4:]
        waveform, sampling_rate = audio_io.audio_loader("{}".format(audio_file))
        # sampling rate가 16000가 아니면 에러 메시지를 띄워줄 수 있도록 함
        assert (
                sampling_rate == 16000
        ), "sampling rate is not consistent throughout the dataset"

        # waveform = self.vad(waveform)
        # waveform = torch.flip(waveform, [0, 1])
        # waveform = self.vad(waveform)
        # waveform = torch.flip(waveform, [0, 1])

        while True:
            audio_length = waveform.shape[1]
            if audio_length < self.audio_window:
                waveform = torch.cat((waveform, waveform), 1)
            else:
                break

        audio_length = waveform.shape[1]
        random_index = np.random.randint(audio_length - self.audio_window + 1)
        waveform = waveform[:, random_index: random_index + self.audio_window]

        mfcc = self.mfcc_transform(waveform)
        return waveform, mfcc, speaker_id
