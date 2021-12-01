import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
from src.utils import file_io_interface
torchaudio.set_audio_backend("sox_io")


def audio_loader(audio_file):
    return torchaudio.load(audio_file)


class WaveformDataset(Dataset):
    def __init__(self, directory_path, audio_window=20480, use_cuda=False):
        self.directory_path = directory_path
        self.audio_window = audio_window
        self.use_cuda = use_cuda

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
        waveform, sampling_rate = audio_loader("{}".format(audio_file))
        # sampling rate가 16000가 아니면 에러 메시지를 띄워줄 수 있도록 함
        assert (
            sampling_rate == 16000
        ), "sampling rate is not consistent throughout the dataset"
        audio_length = waveform.shape[1]
        random_index = np.random.randint(audio_length - self.audio_window + 1)
        waveform = waveform[:, random_index: random_index+self.audio_window]


        return waveform


class SpeakerClassificationDataset(WaveformDataset):
    def __init__(self, speaker_index_file, directory_path, audio_window=20480):
        super(WaveformDataset, self).__init__()
        WaveformDataset.__init__(self, directory_path=directory_path, audio_window=audio_window)
        self.speaker2index = {}
        speaker_data = open(speaker_index_file, 'r')
        speaker_list = [x.strip() for x in speaker_data]
        for i in speaker_list:
            self.speaker2index[i.split(' ')[0]] = int(i.split(' ')[1])

    def __getitem__(self, index):
        item = WaveformDataset.__getitem__(self, index=index)
        audio_file = self.file_list[index]
        audio_file_name = file_io_interface.get_pure_filename(audio_file)
        label = torch.tensor(self.speaker2index[audio_file_name.split('-')[0]])
        return item, label


def get_dataloader(config, mode='train'):
    dataset = None
    dataloader = None
    if config['dataset_type'] == 'WaveformDataset':
        dataset = WaveformDataset(
            directory_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            use_cuda=config['use_cuda'],
        )
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            shuffle=config['dataset_shuffle'],
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
        )
    return dataloader



def get_dataloader_type_direct(directory_path, audio_window, batch_size, num_workers, shuffle, pin_memory):
    dataset = DirectWaveformDataset(directory_path=directory_path, audio_window=audio_window)
    temp = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return temp


def get_dataloader_speaker_classification(directory_path, audio_window,
                                          batch_size, num_workers, shuffle, pin_memory, speaker_index_file):
    dataset = SpeakerClassificationDataset(speaker_index_file=speaker_index_file,
                                           directory_path=directory_path,
                                           audio_window=audio_window)
    temp = data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=pin_memory
    )
    return temp


if __name__ == '__main__':
    # temp_id = open('../dataset/test-librispeech.txt', 'r')
    # sample = temp_id.readline().strip()
    # print(sample)
    # file = h5py.File('../dataset/test-librispeech.h5', 'r')
    # data = torch.tensor(file[sample])
    # print(len(data[0]))
    # print(len(data[0, 0:20480]))

    get_dataloader_speaker_classification(
        directory_path='../../dataset/test-list-librispeech.txt',
        audio_window=20480,
        batch_size=8,
        num_workers=8,
        shuffle=True,
        pin_memory=False,
        speaker_index_file='../../dataset/test-speaker-list-librispeech.txt'
    )



