import torchaudio
import torch
import h5py
from torch.utils.data import Dataset


class WaveformDataset(Dataset):
    def __init__(self, hdf5_file, id_file):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


if __name__ == '__main__':
    file = open('../dataset/test-librispeech.txt', 'r')
    line = file.readline()
    print(line)