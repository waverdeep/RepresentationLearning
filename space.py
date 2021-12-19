import pickle
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torch
import librosa
import numpy as np


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)


if __name__ == '__main__':
    speaker_list_file = open("./dataset/speaker_recognition-train-20480.txt", 'r')
    speaker_file_list = [x.strip() for x in speaker_list_file.readlines()]

    n_fft = 2048
    win_length = None
    hop_length = 512
    n_mels = 256
    n_mfcc = 256
    waveform, sample_rate = torchaudio.load(speaker_file_list[0][4:])
    waveform = waveform[:, :20480]
    print(waveform.size())
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': n_fft,
            'n_mels': n_mels,
            'hop_length': hop_length,
        }
    )

    mfcc = mfcc_transform(waveform)
    plot_spectrogram(mfcc[0])

    vad = T.Vad(16000)
    waveform, sample_rate = torchaudio.load(speaker_file_list[0][4:])
    waveform = vad(waveform)
    waveform = torch.flip(waveform, [0, 1])
    waveform = vad(waveform)
    waveform = torch.flip(waveform, [0, 1])
    while True:
        audio_length = waveform.shape[1]
        if audio_length < 20840:
            waveform = torch.cat((waveform, waveform), 1)
        else:
            break
    print(waveform.size())
    audio_length = waveform.shape[1]
    random_index = np.random.randint(audio_length - 20840 + 1)
    waveform = waveform[:, random_index: random_index + 20840]
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': n_fft,
            'n_mels': n_mels,
            'hop_length': hop_length,
            'mel_scale': 'htk',
        }
    )

    mfcc = mfcc_transform(waveform)
    print(mfcc.size())
    plot_spectrogram(mfcc[0])