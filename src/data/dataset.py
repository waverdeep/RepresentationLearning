import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
from src.utils import file_io_interface
torchaudio.set_audio_backend("sox_io")


def audio_loader(audio_file):
    return torchaudio.load(audio_file)


def get_speaker_list(file_list):
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


class WaveformDataset(Dataset):
    def __init__(self, directory_path, audio_window=20480):
        self.directory_path = directory_path
        self.audio_window = audio_window

        self.file_list = []
        id_data = open(self.directory_path, 'r')
        # strip() 함수를 사용해서 뒤에 개행을 제거
        self.file_list = [x.strip() for x in id_data.readlines()]
        id_data.close()
        self.speaker_list = get_speaker_list(self.file_list)
        self.speaker_dict = get_speaker_dict(self.speaker_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_file = self.file_list[index]
        audio_file = audio_file[4:]
        waveform, sampling_rate = audio_loader("{}".format(audio_file))
        filename = audio_file.split("/")[-1]
        filename = filename.split(".")[0] # speaker_id, dir_id, sample_id
        speaker_id = filename.split("-")[0]

        # sampling rate가 16000가 아니면 에러 메시지를 띄워줄 수 있도록 함
        assert (
            sampling_rate == 16000
        ), "sampling rate is not consistent throughout the dataset"
        audio_length = waveform.shape[1]
        random_index = np.random.randint(audio_length - self.audio_window + 1)
        waveform = waveform[:, random_index: random_index+self.audio_window]
        return waveform, str(filename), str(speaker_id)


def get_dataloader(config, mode='train'):
    dataset = None
    dataloader = None
    if config['dataset_type'] == 'WaveformDataset':
        dataset = WaveformDataset(
            directory_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
        )
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            shuffle=config['dataset_shuffle'],
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
        )
    return dataloader, dataset


if __name__ == '__main__':
    config = {
        "audio_window": 20480,
        "batch_size": 1,
        # dataset
        "dataset_type": "WaveformDataset",
        "train_dataset": "../../dataset/baseline-train-split.txt",
        "test_dataset": "../../dataset/baseline-test-split.txt",
        "num_workers": 16,
        "dataset_shuffle": True,
        "pin_memory": True,
    }
    train_loader, train_dataset = get_dataloader(config=config, mode='train')
    for data in train_loader:
        _, out_filename, speaker_id = data
        print(out_filename)
        print(speaker_id)
        break
    speaker_id_dict = {}
    print(len(list(set(train_dataset.speaker_list))))
    for idx, key in enumerate(sorted(list(set(train_dataset.speaker_list)))):
        speaker_id_dict[key] = idx
    print(speaker_id_dict)

    # get_dataloader_speaker_classification(
    #     directory_path='../../dataset/test-list-librispeech.txt',
    #     audio_window=20480,
    #     batch_size=8,
    #     num_workers=8,
    #     shuffle=True,
    #     pin_memory=False,
    #     speaker_index_file='../../dataset/test-speaker-list-librispeech.txt'
    # )



