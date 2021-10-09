import torchaudio
import torch
import h5py
from torch.utils.data import Dataset



class WaveformDataset(Dataset):
    def __init__(self, hdf5_file, id_file, audio_window):
        self.hdf5_file = h5py.File(hdf5_file)
        self.audio_window = audio_window
        self.audio_id_list = []

        id_data = open(id_file, 'r')
        # strip() 함수를 사용해서 뒤에 개행을 제거
        id_list = [x.strip() for x in id_data.readlines()]
        id_data.close()

        # We train on sampled audio windows of length 20480
        # 논문상 학습을 시킬때 20480 길이로 잘라서 학습을 진행하였다고 함. 그래서 음원의 길이가 20480보다 짧으면 리스트에서 없애버
        for data in id_list:
            temp = self.hdf5_file[data].shape[1]
            if temp > audio_window:
                self.audio_id_list.append(data)

    def __len__(self):
        return len(self.audio_id_list)

    def __getitem__(self, index):
        # We train on sampled audio windows of length 20480
        audio_id = self.audio_id_list[index]
        audio_length = self.hdf5_file[audio_id].shape[1]
        random_index = torch.randint(audio_length - self.audio_window + 1)
        item = torch.tensor(self.hdf5_file[audio_id])
        return item[0, random_index:random_index + self.audio_window]


if __name__ == '__main__':
    temp_id = open('../dataset/test-librispeech.txt', 'r')
    sample = temp_id.readline().strip()
    print(sample)
    file = h5py.File('../dataset/test-librispeech.h5', 'r')
    data = torch.tensor(file[sample])
    print(len(data[0]))
    print(len(data[0, 0:20480]))



