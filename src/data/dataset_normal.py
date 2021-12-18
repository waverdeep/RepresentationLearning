from torch.utils.data import Dataset
import src.utils.interface_audio_io as audio_io


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
            waveform = audio_io.random_cutoff(waveform, self.audio_window)
        return waveform, 0, 0