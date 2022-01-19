import torchaudio
import src.utils.interface_file_io as file_io
import src.data.dataset_baseline as dataset_baseline
import natsort
torchaudio.set_audio_backend("sox_io")


def get_speaker_dict(speaker_list):
    speaker_id_dict = {}
    for idx, key in enumerate(sorted(list(set(speaker_list)))):
        speaker_id_dict[key] = idx
    return speaker_id_dict


def get_audio_file_with_speaker_info(file_list, index):
    audio_file = dataset_baseline.get_audio_file(file_list, index)
    filename = audio_file.split("/")[-1]
    filename = filename.split(".")[0]  # speaker_id, dir_id, sample_id
    speaker_id = filename.split("-")[0]
    return audio_file, filename, speaker_id


# training CPC pretext model
class LibriSpeechWaveformDataset(dataset_baseline.BaselineWaveformDataset):
    def __init__(self, directory_path, audio_window=20480, sample_rate=16000, full_audio=False,
                 augmentation=False, speaker_filelist=None):
        super().__init__(directory_path, audio_window, sample_rate, full_audio, augmentation)
        self.speaker_list = natsort.natsorted(file_io.read_txt2list(speaker_filelist))
        self.speaker_dict = get_speaker_dict(self.speaker_list)

    def __getitem__(self, index):
        audio_file, filename, speaker_id = get_audio_file_with_speaker_info(self.file_list, index)
        waveform = dataset_baseline.load_data_pipeline(audio_file, required_sample_rate=self.sample_rate,
                                                       audio_window=self.audio_window, full_audio=self.full_audio,
                                                       augmentation=self.augmentation)
        return waveform, str(speaker_id)


class LibriSpeechWaveformDatasetByBYOL(LibriSpeechWaveformDataset):
    def __getitem__(self, index):
        audio_file, filename, speaker_id = get_audio_file_with_speaker_info(self.file_list, index)
        aug01_waveform, aug02_waveform = dataset_baseline.load_data_pipeline(audio_file,
                                                                             required_sample_rate=self.sample_rate,
                                                                             audio_window=self.audio_window,
                                                                             full_audio=self.full_audio,
                                                                             augmentation=self.augmentation)
        return aug01_waveform, aug02_waveform, str(filename), str(speaker_id)
