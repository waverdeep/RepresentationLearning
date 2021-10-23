from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import src.utils.file_io_interface as io
import numpy as np


def get_speaker_list(directory_path_list, new_filepath_without_extension, audio_window):
    train_audio_file = open('{}-audio-train.txt'.format(new_filepath_without_extension), 'w')
    test_audio_file = open('{}-audio-test.txt'.format(new_filepath_without_extension), 'w')
    train_audio_id_file = open('{}-train.txt'.format(new_filepath_without_extension), 'w')
    test_audio_id_file = open('{}-test.txt'.format(new_filepath_without_extension), 'w')
    audio_id_file = open('{}-id_fille.txt'.format(new_filepath_without_extension), 'w')
    audio_file_list = []
    audio_id_list = []
    for directory in directory_path_list:
        file_list = io.get_all_file_path(directory, 'flac')
        for file in tqdm(file_list, desc=directory):
            waveform, sampling_rate = torchaudio.load(file)
            if waveform.shape[1] > audio_window:
                audio_file_list.append(file)
                filename = io.get_pure_filename(file)
                audio_id_list.append(filename.split('-')[0])

    speaker_list = list(set(audio_id_list))
    print('speaker_list: ', len(speaker_list))

    for index, speaker_id in enumerate(speaker_list):
        audio_id_file.write('{} {}\n'.format(speaker_id, index))
    audio_id_file.close()

    train_audio, test_audio = train_test_split(audio_file_list, shuffle=True, test_size=0.3)

    for index, audio in enumerate(train_audio):
        train_audio_file.write('{}\n'.format(audio))
    train_audio_file.close()

    for index, audio in enumerate(test_audio):
        test_audio_file.write('{}\n'.format(audio))
    test_audio_file.close()

    for index, audio in enumerate(train_audio):
        filename = io.get_pure_filename(audio)
        speaker_id = filename.split('-')[0]
        speaker_index = speaker_list.index(speaker_id)
        train_audio_id_file.write('{} {}\n'.format(speaker_id, speaker_index))
    train_audio_id_file.close()

    for index, audio in enumerate(test_audio):
        filename = io.get_pure_filename(audio)
        speaker_id = filename.split('-')[0]
        speaker_index = speaker_list.index(speaker_id)
        test_audio_id_file.write('{} {}\n'.format(speaker_id, speaker_index))
    test_audio_id_file.close()


if __name__ == '__main__':
    train_directory_path = ['../../dataset/LibriSpeech/train-clean-100']

    get_speaker_list(train_directory_path, '../../dataset/speaker-list-librispeech', 20480)
