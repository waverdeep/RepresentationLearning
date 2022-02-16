import src.data.dataset_baseline as dataset_baseline
import src.data.dataset_librispeech as dataset_librispeech
import src.utils.interface_file_io as file_io
import natsort


def get_audio_file_with_speaker_info(file_list, index):
    audio_file = dataset_baseline.get_audio_file(file_list, index)
    temp = audio_file.split('/')
    speaker_id = temp[4]
    return audio_file, speaker_id


class VoxCelebWaveformDataset(dataset_baseline.BaselineWaveformDataset):
    def __init__(self, file_path, audio_window=20480, sample_rate=16000, full_audio=False,
                 augmentation=False, speaker_filelist="./dataset/voxceleb01-SI-label.txt"):
        super().__init__(file_path=file_path, audio_window=audio_window, sample_rate=sample_rate,
                         full_audio=full_audio, augmentation=augmentation)
        self.speaker_list = natsort.natsorted(file_io.read_txt2list(speaker_filelist))
        self.speaker_dict = dataset_librispeech.get_speaker_dict(self.speaker_list)

    def __getitem__(self, index):
        # ../../dataset/vox01/wav/id10977/radm0JQM9aI/00012.wav
        audio_file, speaker_id = get_audio_file_with_speaker_info(self.file_list, index)
        waveform = dataset_baseline.load_data_pipeline(audio_file, required_sample_rate=self.sample_rate,
                                                       audio_window=self.audio_window, full_audio=self.full_audio,
                                                       augmentation=self.augmentation, custom_augmentation_list=[0, 2, 3, 6])
        return waveform, speaker_id