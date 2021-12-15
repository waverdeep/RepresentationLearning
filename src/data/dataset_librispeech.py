import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
import src.utils.interface_audio_io as audio_io
import src.data.dataset_tool_speaker as speaker_tool
torchaudio.set_audio_backend("sox_io")


class LibriSpeechWaveformDataset(Dataset):
    def __init__(self, directory_path, audio_window=20480, sampling_rate=16000, auto_trim=False, full_audio=False):
        self.directory_path = directory_path
        self.audio_window = audio_window
        self.sampling_rate = sampling_rate
        self.auto_trim = auto_trim
        self.full_audio = full_audio

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
            waveform = speaker_tool.audio_auto_trim(waveform, self.vad, self.audio_window)

        if not self.full_audio:
            waveform = speaker_tool.random_cutoff(waveform, self.audio_window)
        return waveform, str(filename), str(speaker_id)

    def get_audio_by_speaker(self, speaker_id, batch_size):
        batch_size = min(len(self.speaker_align[speaker_id]), batch_size)
        batch = torch.zeros(batch_size, 1, self.audio_window)
        for index in range(batch_size):
            batch[index, 0, :], _, _ = self.__getitem__(self.speaker_align[speaker_id][index])
        return batch