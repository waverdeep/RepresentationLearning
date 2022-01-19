import src.data.dataset_baseline as dataset_baseline
import src.data.dataset_librispeech as dataset_librispeech
import src.utils.interface_file_io as file_io
import natsort


def get_speaker_list(file_list):
    speaker_list = []
    for index, file in enumerate(file_list):
        temp = file.split('/')
        speaker_id = temp[5][2:]
        speaker_list.append(speaker_id)
    return speaker_list


def get_audio_file_with_speaker_info(file_list, index):
    audio_file = dataset_baseline.get_audio_file(file_list, index)
    temp = audio_file.split('/')
    speaker_id = temp[5][2:]
    return audio_file, speaker_id


class VoxWaveformDataset(dataset_baseline.BaselineWaveformDataset):
    def __init__(self, directory_path, audio_window=20480, sample_rate=16000, full_audio=False,
                 augmentation=False, speaker_filelist=None):
        super().__init__(directory_path, audio_window, sample_rate, full_audio, augmentation)
        self.speaker_list = natsort.natsorted(file_io.read_txt2list(speaker_filelist))
        self.speaker_dict = dataset_librispeech.get_speaker_dict(self.speaker_list)

    def __getitem__(self, index):
        # ../../dataset/vox01/wav/id10977/radm0JQM9aI/00012.wav
        audio_file, speaker_id = get_audio_file_with_speaker_info(self.file_list, index)
        waveform = dataset_baseline.load_data_pipeline(audio_file, required_sample_rate=self.sample_rate,
                                                       audio_window=self.audio_window, full_audio=self.full_audio,
                                                       augmentation=self.augmentation)
        return waveform, speaker_id