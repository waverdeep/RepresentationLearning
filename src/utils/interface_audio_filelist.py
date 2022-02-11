import torchaudio
from tqdm import tqdm
import src.utils.interface_file_io as io


def get_audio_list(directory_path_list, new_filepath, audio_window, file_extension="flac", skip_check=True):
    audio_list_file = open('{}'.format(new_filepath), 'w')
    for directory in directory_path_list:
        file_list = io.get_all_file_path(directory, file_extension)
        for file in tqdm(file_list, desc=directory):
            if not skip_check:
                waveform, sampling_rate = torchaudio.load(file)
                if waveform.shape[1] > audio_window:
                    # filename = io.get_pure_filename(file)s
                    audio_list_file.write('{}\n'.format(file))
            else:
                audio_list_file.write('{}\n'.format(file))
    audio_list_file.close()


def get_baseline_audio_list(directory_path, original_filepath, new_filepath):
    audio_list_file = open('{}'.format(new_filepath), 'w')
    id_data = open(original_filepath, 'r')
    file_list = [x.strip() for x in id_data.readlines()]
    id_data.close()
    for index, file in enumerate(file_list):
        path = file.split('-')
        audio_list_file.write('{}/{}/{}/{}.flac\n'.format(directory_path, path[0], path[1], file))
    audio_list_file.close()


if __name__ == '__main__':
    name = "vox01"
    if name == "custom_dataset":
        train_directory_path = ['../../dataset/LibriSpeech/train-clean-100', '../../dataset/LibriSpeech/train-clean-360',
                                '../../dataset/LibriSpeech/train-other-500']
        dev_directory_path = ['../../dataset/LibriSpeech/dev-clean', '../../dataset/LibriSpeech/dev-other']
        test_directory_path = ['../../dataset/LibriSpeech/test-clean', '../../dataset/LibriSpeech/test-other']

        get_audio_list(train_directory_path, '../../dataset/train-list-librispeech-32000.txt', audio_window=20480)
        get_audio_list(dev_directory_path, '../../dataset/dev-list-librispeech-32000.txt', audio_window=20480)
        get_audio_list(test_directory_path, '../../dataset/test-list-librispeech-32000.txt', audio_window=20480)
    elif name == "original dataset":
        # original dataset
        train_directory_path = '../../dataset/LibriSpeech/train-clean-100'
        test_directory_path = '../../dataset/LibriSpeech/train-clean-100'

        get_baseline_audio_list(train_directory_path, '../../dataset/train_split.txt',
                                '../../dataset/baseline-train-split.txt')
        get_baseline_audio_list(test_directory_path, '../../dataset/test_split.txt',
                                '../../dataset/baseline-test-split.txt')
    elif name == "urban_sound":
        directory_path = ['../../dataset/UrbanSound8K/audio_16k']
        get_audio_list(directory_path, '../../dataset/urbansound8k.txt', audio_window=20480, file_extension="wav")
    elif name == "kspon":
        directory_path = ['../../dataset/KsponSpeech/train']
        get_audio_list(directory_path, '../../dataset/kspon-20480.txt', audio_window=20480, file_extension="wav")
    elif name == "librispeech360":
        train_directory_path = ['../../dataset/LibriSpeech/train-clean-360']
        get_audio_list(train_directory_path, '../../dataset/librispeech360-20480.txt', audio_window=20480)
    elif name == 'vox01':
        train_directory_path = ['../../dataset/voxceleb/wav']
        get_audio_list(train_directory_path, '../../dataset/voxceleb01-20480.txt', audio_window=20480, file_extension="wav")
    elif name == 'competition':
        train_directory_path = ['../../dataset/speaker_recognition/train']
        get_audio_list(train_directory_path, '../../dataset/speaker_recognition-train-20480.txt', audio_window=20480,
                       file_extension="wav")
        train_directory_path = ['../../dataset/speaker_recognition/dev']
        get_audio_list(train_directory_path, '../../dataset/speaker_recognition-dev-20480.txt', audio_window=20480,
                       file_extension="wav")
    elif name == 'FSD50K.dev_audio':
        train_directory_path = ['../../dataset/FSD50K.dev_audio_16k']
        get_audio_list(train_directory_path, '../../dataset/FSD50K.txt', audio_window=20480,
                       file_extension="wav")