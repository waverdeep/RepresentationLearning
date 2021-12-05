import soundfile as sf
from tqdm import tqdm
import src.utils.interface_file_io as io
import librosa
import wave
import multiprocessing
import interface_multiprocessing as mi


def resampling_audio(file, original_sampling_rate=44100, resampling_rate=16000):
    waveform, sampling_rate = librosa(file, original_sampling_rate)
    resample_waveform = librosa.resample(waveform, original_sampling_rate, resampling_rate)
    return resample_waveform


def resampling_audio_list(directory_list, new_file_path, file_extension, original_sampling_rate, resampling_rate):
    for dir_index, directory in directory_list:
        file_list = io.get_all_file_path(directory, file_extension=file_extension)
        for file_index, file in tqdm(file_list, desc=directory):
            resample_waveform = resampling_audio(file, original_sampling_rate=original_sampling_rate,
                                                 resampling_rate=resampling_rate)
            filename = io.get_pure_filename(file)
            file_path = "{}/{}".format(new_file_path, filename)
            sf.write(file_path, resample_waveform, resampling_rate)


# The parameters are prerequisite information. More specifically,
# channels, bit_depth, sampling_rate must be known to use this function.
def pcm2wav(pcm_file, wav_file=None, channels=1, bit_depth=16, sampling_rate=16000):
    # Check if the options are valid.
    if bit_depth % 8 != 0:
        raise ValueError("bit_depth " + str(bit_depth) + " must be a multiple of 8.")

    if wav_file is None:
        wav_file = pcm_file.replace("pcm", "wav")

    # Read the .pcm file as a binary file and store the data to pcm_data
    with open(pcm_file, 'rb') as opened_pcm_file:
        pcm_data = opened_pcm_file.read()

        obj2write = wave.open(wav_file, 'wb')
        obj2write.setnchannels(channels)
        obj2write.setsampwidth(bit_depth // 8)
        obj2write.setframerate(sampling_rate)
        obj2write.writeframes(pcm_data)
        obj2write.close()


def distributed_pcm2wav(pcm_file):
    print("start data distribution...")
    for pcm_index, pcm in enumerate(pcm_file):
        pcm2wav(pcm)
    print("end data distribution...")


if __name__ == '__main__':
    task = ""
    if task == "resampling":
        directory_path = ['../../dataset/UrbanSound8K/audio']
        new_save_directory = '../../dataset/UrbanSound8K/audio_16k/'
        resampling_audio_list(directory_path, new_save_directory, 'wav', 44100, 16000)

    elif task == 'pcm2wav':
        input_dir = "../../dataset/KsponSpeech/train"
        file_extension = "pcm"
        divide_num = multiprocessing.cpu_count() - 1

        file_list = io.get_all_file_path(input_dir, file_extension)
        file_list = io.list_divider(divide_num, file_list)
        print(len(file_list))

        processes = mi.setup_multiproceesing(distributed_pcm2wav, data_list=file_list)
        mi.start_multiprocessing(processes)
