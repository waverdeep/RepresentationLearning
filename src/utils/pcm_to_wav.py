# thanks to
# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=aimldl&logNo=221559323232
import wave
from multiprocessing import Process
import multiprocessing
import src.utils.file_io_interface as io


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
    for pcm_index, pcm in enumerate(pcm_file):
        pcm2wav(pcm)


def list_divider(step, data):
    split_len = int(len(data)/step)
    return [data[i:i+split_len] for i in range(0, len(data), split_len)]


if __name__ == '__main__':
    input_dir = "../../dataset/KsponSpeech/train"
    file_extension = "pcm"
    divide_num = multiprocessing.cpu_count() - 1
    processes = []

    file_list = io.get_all_file_path(input_dir, file_extension)
    file_list = list_divider(divide_num, file_list)
    print(len(file_list))

    for index, file in enumerate(file_list):
        process = Process(target=distributed_pcm2wav, args=(file,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
