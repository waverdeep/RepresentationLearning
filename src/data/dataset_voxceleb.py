import src.data.dataset_tool_speaker as speaker_tool
import src.utils.interface_audio_io as audio_io
import src.data.dataset_normal as normal


def get_vox_speaker_list(file_list):
    speaker_list = []
    for index, file in enumerate(file_list):
        temp = file.split('/')
        speaker_id = temp[5][2:]
        speaker_list.append(speaker_id)
    return speaker_list


class VoxWaveformDataset(normal.NormalWaveformDataset):
    def __init__(self, directory_path, audio_window=20480, augmentation=False, full_audio=False):
        normal.NormalWaveformDataset.__init__(self, directory_path, audio_window)
        self.audio_window = audio_window
        self.full_audio = full_audio
        self.augmentation = augmentation
        self.speaker_list = get_vox_speaker_list(self.file_list)
        self.speaker_dict = speaker_tool.get_speaker_dict(self.speaker_list)

    def __getitem__(self, index):
        # ../../dataset/vox01/wav/id10977/radm0JQM9aI/00012.wav
        audio_file = self.file_list[index]
        temp = audio_file.split('/')
        speaker_id = temp[5][2:]
        audio_file = audio_file[4:]
        waveform, sampling_rate = audio_io.audio_loader("{}".format(audio_file))
        # sampling rate가 16000가 아니면 에러 메시지를 띄워줄 수 있도록 함
        assert (
                sampling_rate == 16000
        ), "sampling rate is not consistent throughout the dataset"
        if not self.full_audio:
            waveform = audio_io.random_cutoff(waveform, self.audio_window)
        return waveform, 0, speaker_id