import torchaudio
import h5py
import glob
import os
from tqdm import tqdm

'''
Hierachical Data Format version 5
 - easy sharing
 - cross platform
 - fast IO
 - Big data
 - Heterogeneous data
대용량 데이터를 저장하기 위한 파일 포맷임 (속도가 빠르다고 함) 
'''


def get_pure_filename(filename):
    temp = filename.split('.')
    del temp[-1]
    temp = '.'.join(temp)
    temp = temp.split('/')
    temp = temp[-1]
    return temp


def get_all_file_path(input_dir, file_extension):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp


def wav_to_hdf5(directory_path_list, new_filepath_without_extension):
    h5_file = h5py.File('{}.h5'.format(new_filepath_without_extension), 'w')
    id_file = open('{}.txt'.format(new_filepath_without_extension), 'w')
    for directory_path in directory_path_list:
        file_list = get_all_file_path(directory_path, 'flac')
        for file in tqdm(file_list, desc=directory_path):
            waveform, sampling_rate = torchaudio.load(file)
            filename = get_pure_filename(file)
            h5_file.create_dataset(filename, data=waveform)
            id_file.write('{}\n'.format(filename))
    h5_file.close()
    id_file.close()


if __name__ == '__main__':
    train_directory_path = ['../dataset/LibriSpeech/train-clean-100', '../dataset/LibriSpeech/train-clean-360',
                            '../dataset/LibriSpeech/train-clean-500']
    dev_directory_path = ['../dataset/LibriSpeech/dev-clean', '../dataset/LibriSpeech/dev-other']
    test_directory_path = ['../dataset/LibriSpeech/test-clean', '../dataset/LibriSpeech/test-other']
    tiny_train_directory_path = ['../dataset/LibriSpeech/tiny-train']
    tiny_dev_directory_path = ['../dataset/LibriSpeech/tiny-dev']

    # wav_to_hdf5(train_directory_path, '../dataset/train-librispeech')
    # wav_to_hdf5(dev_directory_path, '../dataset/dev-librispeech')
    # wav_to_hdf5(test_directory_path, '../dataset/test-librispeech')
    wav_to_hdf5(tiny_train_directory_path, '../dataset/tiny-train-librispeech')
    wav_to_hdf5(tiny_dev_directory_path, '../dataset/tiny-dev-librispeech')







