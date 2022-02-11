import os

import soundfile as sf
from tqdm import tqdm
import src.utils.interface_file_io as io

import librosa
import multiprocessing
import src.utils.interface_multiprocessing as mi


def get_audio_list(directory_path_list, new_save_path='../../dataset/FSD50K.dev_audio_16k/', audio_window=20480, file_extension="wav"):
    for directory in directory_path_list:
        file_list = io.get_all_file_path(directory, file_extension)
        for file in tqdm(file_list, desc=directory):
            waveform, sampling_rate = librosa.load(file, 44100)
            resample_waveform = librosa.resample(waveform, 44100, 16000)
            new_filename = file.replace("audio", "audio_16k")
            sf.write(new_filename, resample_waveform, 16000)


def resampling_audio(file_list):
    list_size = len(file_list)
    for index, file in enumerate(file_list):
        waveform, sampling_rate = librosa.load(file, 44100)
        resample_waveform = librosa.resample(waveform, 44100, 16000)
        new_filename = file.replace("audio", "audio_16k")
        sf.write(new_filename, resample_waveform, 16000)
        if index % 50 == 0:
            proc = os.getpid()
            print("P{}: {}/{}".format(proc, index, list_size))


def aaa():
    directory_path = '../../dataset/FSD50K.dev_audio'
    file_extension = "wav"
    divide_num = multiprocessing.cpu_count() - 1
    print(divide_num)
    file_list = io.get_all_file_path(directory_path, file_extension)
    file_list = io.list_divider(divide_num, file_list)
    print(len(file_list))

    processes = mi.setup_multiproceesing(resampling_audio, data_list=file_list)
    mi.start_multiprocessing(processes)


if __name__ == '__main__':
    aaa()
    # resampling_audio()
    # directory_path = ['../../dataset/FSD50K.dev_audio']
    # get_audio_list(directory_path, '../../dataset/FSD50K.dev_audio_16k/', audio_window=20480, file_extension="wav")

    # directory_path = ['../../dataset/UrbanSound8K/audio']
    # get_audio_list(directory_path, '../../dataset/UrbanSound8K/audio_16k/', audio_window=20480, file_extension="wav")
