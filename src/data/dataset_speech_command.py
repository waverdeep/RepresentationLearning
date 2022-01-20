import torchaudio.datasets as datasets
import natsort

import src.utils.interface_file_io as file_io
import src.data.dataset_librispeech as dataset_librispeech


class SpeechCommandWaveformDataset(datasets.SPEECHCOMMANDS):
    def __init__(self, root, download, command_filelist):
        super().__init__(root=root, download=download)
        self.command_list = natsort.natsorted(file_io.read_txt2_list(command_filelist))
        self.command_dict = dataset_librispeech.get_speaker_dict(self.command_list)

    def __getitem__(self, n):
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(n)
        return waveform, label
