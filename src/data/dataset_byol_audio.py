import src.data.dataset_normal as normal
import src.utils.interface_audio_io as audio_io
import src.utils.interface_audio_augmentation as audio_augmentation
import torch.nn.functional as F
import random
import torch


class ByolAudioDataset(normal.NormalWaveformDataset):
    def __init__(self, directory_path, audio_window=20480, full_audio=False, config=None, use_librosa=True,
                 mode='train'):
        super().__init__(directory_path=directory_path, audio_window=audio_window, full_audio=full_audio)
        self.config = config
        if mode == 'train':
            self.transforms = audio_augmentation.AugmentationModule((64, 96), 2 * len(self.file_list))
        else:
            self.transforms = audio_augmentation.AugmentationModule((64, 96), 2 * len(self.file_list))
        self.to_melspectrogram = audio_io.MelSpectrogramLibrosa(
            fs=config['sampling_rate'],
            n_fft=config['n_fft'],
            shift=config['hop_length'],
            n_mels=config['n_mels'],
            fmin=config['f_min'],
            fmax=config['f_max'],
        )

    def __getitem__(self, index):
        audio_file = self.file_list[index]
        audio_file = audio_file[4:]
        waveform, sampling_rate = audio_io.audio_loader("{}".format(audio_file))
        # sampling rate가 16000가 아니면 에러 메시지를 띄워줄 수 있도록 함
        assert (
                sampling_rate == 16000
        ), "sampling rate is not consistent throughout the dataset"

        waveform = waveform[0]  # (1, length) -> (length,)

        # zero padding to both ends
        length_adj = self.audio_window - len(waveform)
        if length_adj > 0:
            half_adj = length_adj // 2
            waveform = F.pad(waveform, (half_adj, length_adj - half_adj))

        # random crop unit length wave
        length_adj = len(waveform) - self.audio_window
        start = random.randint(0, length_adj) if length_adj > 0 else 0
        waveform = waveform[start:start + self.audio_window]

        # to log mel spectrogram -> (1, n_mels, time)
        log_mel_spectrogram = (self.to_melspectrogram(waveform) + torch.finfo().eps).log().unsqueeze(0)

        # transform (augment)
        if self.transforms:
            log_mel_spectrogram = self.transforms(log_mel_spectrogram)
        else:
            log_mel_spectrogram = (log_mel_spectrogram, log_mel_spectrogram)

        return log_mel_spectrogram, 0, 0
