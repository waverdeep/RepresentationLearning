import src.data.dataset_baseline as dataset_baseline
import src.utils.interface_file_io as file_io

import pandas as pd
import natsort


def get_acoustic_dict(acoustic_list):
    acoustic_dict = {}
    for idx, key in enumerate(acoustic_list):
        acoustic_dict[str(key)] = idx
    return acoustic_dict


def get_audio_file_with_acoustic_info(file_list, index):
    audio_file = dataset_baseline.get_audio_file(file_list, index)
    filename = audio_file.split('/')[5]
    acoustic_id = filename.split('-')[1]
    return audio_file, filename, acoustic_id


def search_cutting_boundary(metadata, filename):
    line = metadata[metadata['slice_file_name'] == filename]
    bound = [float(line['start']), float(line['end'])]
    return bound


# num_classes = 10
class UrbanSound8KWaveformDataset(dataset_baseline.BaselineWaveformDataset):
    def __init__(self, file_path: str, audio_window=20480, sample_rate=16000, full_audio=False, augmentation=False,
                 metadata=None):
        super().__init__(file_path=file_path, audio_window=audio_window, sample_rate=sample_rate, full_audio=full_audio,
                         augmentation=augmentation)
        self.metadata = None
        if metadata is not None:
            self.metadata = pd.read_csv(metadata)
        self.acoustic_list = natsort.natsorted(list(set(self.metadata['classID'])))
        self.acoustic_dict = get_acoustic_dict(self.acoustic_list)

    def __getitem__(self, index):
        audio_file, filename, acoustic_id = get_audio_file_with_acoustic_info(self.file_list, index)
        cutting_bound = None
        if self.metadata is not None:
            cutting_bound = search_cutting_boundary(self.metadata, filename)
        waveform = dataset_baseline.load_data_pipeline(audio_file, required_sample_rate=self.sample_rate,
                                                       audio_window=self.audio_window, full_audio=self.full_audio,
                                                       augmentation=self.augmentation, cut_silence=cutting_bound)
        return waveform, str(acoustic_id)

