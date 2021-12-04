import soundfile as sf
from tqdm import tqdm
import src.utils.file_io_interface as io
import librosa


def get_audio_list(directory_path_list, new_save_path, audio_window, file_extension="flac"):
    for directory in directory_path_list:
        file_list = io.get_all_file_path(directory, file_extension)
        for file in tqdm(file_list, desc=directory):
            waveform, sampling_rate = librosa.load(file, 44100)
            resample_waveform = librosa.resample(waveform, 44100, 16000)
            new_filename = file.replace("audio", "audio_16k")
            sf.write(new_filename, resample_waveform, 16000)


if __name__ == '__main__':
    directory_path = ['../../dataset/UrbanSound8K/audio']
    get_audio_list(directory_path, '../../dataset/UrbanSound8K/audio_16k/', audio_window=20480, file_extension="wav")
