import pickle
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torch
import librosa
import numpy as np
import json
import src.models.model_proposed01 as model_proposed
import src.utils.interface_file_io as file_io
import src.utils.interface_audio_augmentation as audio_augmentation
import torchaudio
import torchaudio.datasets as datasets
from torch.utils import data


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
    # print(len(file_io.read_txt2list('./dataset/librispeech100-baseline-test.txt')))
    # sample = torch.rand(1, 20480)
    # audio_augmentation.audio_additive_noise(sample, 16000)
    # datasets.SPEECHCOMMANDS()
    # command_dataset = datasets.SPEECHCOMMANDS(root='./dataset', download=True)
    # command_dataloader = data.DataLoader(command_dataset, shuffle=True, batch_size=2)
    # dataiter = iter(command_dataloader)
    # waveform, sample_rate, label, speaker_id, utterance_number = dataiter.next()
    # print(waveform.size())
    # print(sample_rate)
    # print(label)
    # print(speaker_id)
    # print(utterance_number)
    data = torch.rand(2, 1, 64000).cuda()

