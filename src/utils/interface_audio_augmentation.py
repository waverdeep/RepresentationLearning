import augment
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as AT
import torchaudio.functional as AF
import src.utils.interface_file_io as file_io
import src.utils.interface_audio_io as audio_io
import random


def audio_additive_noise(x, sr, audio_window=20480, datalist_path="./dataset/musan-total.txt"):
    def noise_generator():
        dataset_path = datalist_path
        filelist = file_io.read_txt2list(dataset_path)
        pick = np.random.randint(len(filelist))
        waveform, sampling_rate = audio_io.audio_loader(filelist[pick][4:])
        waveform = audio_io.audio_adjust_length(waveform, audio_window)
        waveform = audio_io.random_cutoff(waveform, audio_window)
        return waveform[0]
    combination = augment.EffectChain() \
        .additive_noise(noise_generator, snr=15)
    y = combination.apply(x, src_info={'rate': sr}, target_info={'rate': sr})
    return y


def audio_band_reject(x, sr, audio_window=None):
    combination = augment.EffectChain().sinc('-a', '120', '500-100')
    y = combination.apply(x, src_info={'rate': sr})
    return y


def audio_time_dropout(x, sr, max_seconds=0.1, audio_window=None):
    combination = augment.EffectChain().time_dropout(max_seconds=max_seconds)
    y = combination.apply(x, src_info={'rate': sr}, target_info={'rate': sr})
    return y


def audio_reverb(x, sr, reverb_size=100, audio_window=None):
    random_room_size = lambda: np.random.randint(0, 101)
    combination = augment.EffectChain().reverb(reverb_size, reverb_size, random_room_size).channels(1)
    y = combination.apply(x, src_info={'rate': sr}, target_info={'rate': sr})
    return y


def audio_pitch_shift(x, sr, shift_size=500, audio_window=None):
    random_pitch_shift = lambda: np.random.randint(-shift_size, +shift_size)
    combination = augment.EffectChain().pitch("-q", random_pitch_shift).rate(sr)
    y = combination.apply(x, src_info={'rate': sr}, target_info={'rate': sr})
    return y


def audio_clipping_audio(x, sr, clipping_rate=0.25):
    combination = augment.EffectChain().clip(clipping_rate)
    y = combination.apply(x, src_info={'rate': sr})
    return y


def audio_speed(x, sr, rate=None):
    if rate is not None:
        effects = [['speed', str(rate)]]
    else:
        random = []
        effects = [['speed', ]]
    torchaudio.sox_effects.apply_effects_tensor(x, sr, effects)


def audio_augmentation_pipeline(x, sr, audio_window, pick_augmentation,fix_audio_length=True):
    pipeline = []
    for pick in pick_augmentation:
        if pick == 0:
            pipeline.append(audio_band_reject)
        elif pick == 1:
            pipeline.append(audio_time_dropout)
        elif pick == 2:
            pipeline.append(audio_reverb)
        elif pick == 3:
            pipeline.append(audio_pitch_shift)
        elif pick == 4:
            pipeline.append(audio_clipping_audio)
        elif pick == 5:
            pipeline.append(audio_additive_noise)

    for method in pipeline:
        x = method(x=x, sr=sr, audio_window=audio_window)
        if fix_audio_length:
            if len(x[0]) != audio_window:
                x = audio_io.audio_adjust_length(x, audio_window, True)
    return x


def audio_augmentation_baseline(x, sr=16000, audio_window=20480, fix_audio_length=False):
    augmentation_list = [0, 2, 3, 5]
    pick_augmentation = random.sample(augmentation_list, 3)
    audio_augmentation_pipeline(x, sr, audio_window, pick_augmentation, fix_audio_length)
    return x


########################################################################################################################
# https://github.com/nttcslab/byol-a
"""BYOL for Audio: Augmentation modules.

Legends:
    F: Number of frequency bins.
    T: Number of time frames.
"""

class RandomResizeCrop(nn.Module):
    """Random Resize Crop block.

    Args:
        virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
        freq_scale: Random frequency range `(min, max)`.
        time_scale: Random time frame range `(min, max)`.
    """

    def __init__(self, virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)):
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = 'bicubic'
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward(self, lms):
        # make virtual_crop_arear empty space (virtual crop area) and copy the input log mel spectrogram to th the center
        virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
        virtual_crop_area = (torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]))
                             .to(torch.float).to(lms.device))
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y:y+h, x:x+w] = lms
        # get random area
        i, j, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], self.time_scale, self.freq_scale)
        crop = virtual_crop_area[:, i:i+h, j:j+w]
        # print(f'shapes {virtual_crop_area.shape} {crop.shape} -> {lms.shape}')
        lms = F.interpolate(crop.unsqueeze(0), size=lms.shape[-2:],
                            mode=self.interpolation, align_corners=True).squeeze(0)
        return lms.to(torch.float)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(virtual_crop_size={self.virtual_crop_scale}'
        format_string += ', time_scale={0}'.format(tuple(round(s, 4) for s in self.time_scale))
        format_string += ', freq_scale={0})'.format(tuple(round(r, 4) for r in self.freq_scale))
        return format_string


def log_mixup_exp(xa, xb, alpha):
    xa = xa.exp()
    xb = xb.exp()
    x = alpha * xa + (1. - alpha) * xb
    return torch.log(x + torch.finfo(x.dtype).eps)


class MixupBYOLA(nn.Module):
    """Mixup for BYOL-A.

    Args:
        ratio: Alpha in the paper.
        n_memory: Size of memory bank FIFO.
        log_mixup_exp: Use log-mixup-exp to mix if this is True, or mix without notion of log-scale.
    """

    def __init__(self, ratio=0.4, n_memory=2048, log_mixup_exp=True):
        super().__init__()
        self.ratio = ratio
        self.n = n_memory
        self.log_mixup_exp = log_mixup_exp
        self.memory_bank = []

    def forward(self, x):
        # mix random
        alpha = self.ratio * np.random.random()
        if self.memory_bank:
            # get z as a mixing background sound
            z = self.memory_bank[np.random.randint(len(self.memory_bank))]
            # mix them
            mixed = log_mixup_exp(x, z, 1. - alpha) if self.log_mixup_exp \
                    else alpha * z + (1. - alpha) * x
        else:
            mixed = x
        # update memory bank
        self.memory_bank = (self.memory_bank + [x])[-self.n:]

        return mixed.to(torch.float)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(ratio={self.ratio},n={self.n}'
        format_string += f',log_mixup_exp={self.log_mixup_exp})'
        return format_string


class MixGaussianNoise():
    """Gaussian Noise Mixer.
    This interpolates with random sample, unlike Mixup.
    """

    def __init__(self, ratio=0.3):
        self.ratio = ratio

    def forward(self, lms):
        x = lms.exp()

        lambd = self.ratio * np.random.rand()
        z = torch.normal(0, lambd, x.shape).exp()
        mixed = (1 - lambd) * x + z + torch.finfo(x.dtype).eps

        return mixed.log()

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(ratio={self.ratio})'
        return format_string


class RunningMean:
    """Running mean calculator for arbitrary axis configuration."""

    def __init__(self, axis):
        self.n = 0
        self.axis = axis

    def put(self, x):
        # https://math.stackexchange.com/questions/106700/incremental-averageing
        if self.n == 0:
            self.mu = x.mean(self.axis, keepdims=True)
        else:
            self.mu += (x.mean(self.axis, keepdims=True) - self.mu) / self.n
        self.n += 1

    def __call__(self):
        return self.mu

    def __len__(self):
        return self.n


class RunningVariance:
    """Calculate mean/variance of tensors online.
    Thanks to https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self, axis, mean):
        self.update_mean(mean)
        self.s2 = RunningMean(axis)

    def update_mean(self, mean):
        self.mean = mean

    def put(self, x):
        self.s2.put((x - self.mean) **2)

    def __call__(self):
        return self.s2()

    def std(self):
        return np.sqrt(self())


class RunningNorm(nn.Module):
    """Online Normalization using Running Mean/Std.

    This module will only update the statistics up to the specified number of epochs.
    After the `max_update_epochs`, this will normalize with the last updated statistics.

    Args:
        epoch_samples: Number of samples in one epoch
        max_update_epochs: Number of epochs to allow update of running mean/variance.
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, epoch_samples, max_update_epochs=10, axis=[1, 2]):
        super().__init__()
        self.max_update = epoch_samples * max_update_epochs
        self.ema_mean = RunningMean(axis)
        self.ema_var = RunningVariance(axis, 0)

    def forward(self, image):
        if len(self.ema_mean) < self.max_update:
            self.ema_mean.put(image)
            self.ema_var.update_mean(self.ema_mean())
            self.ema_var.put(image)
            self.mean = self.ema_mean()
            self.std = torch.clamp(self.ema_var.std(), torch.finfo().eps, torch.finfo().max)
        return ((image - self.mean) / self.std)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(max_update={self.max_update},axis={self.ema_mean.axis})'
        return format_string


class PrecomputedNorm(nn.Module):
    """Normalization using Pre-computed Mean/Std.

    Args:
        stats: Precomputed (mean, std).
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, stats, axis=[1, 2]):
        super().__init__()
        self.axis = axis
        self.mean, self.std = stats

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return ((X - self.mean) / self.std)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(mean={self.mean}, std={self.std}, axis={self.axis})'
        return format_string


class NormalizeBatch(nn.Module):
    """Normalization of Input Batch.

    Note:
        Unlike other blocks, use this with *batch inputs*.

    Args:
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, axis=[0, 2, 3]):
        super().__init__()
        self.axis = axis

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _mean = X.mean(dim=self.axis, keepdims=True)
        _std = torch.clamp(X.std(dim=self.axis, keepdims=True), torch.finfo().eps, torch.finfo().max)
        return ((X - _mean) / _std)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(axis={self.axis})'
        return format_string


class AugmentationModule:
    """BYOL-A augmentation module example, the same parameter with the paper."""

    def __init__(self, size, epoch_samples, log_mixup_exp=True, mixup_ratio=0.4):
        self.train_transform = nn.Sequential(
            MixupBYOLA(ratio=mixup_ratio, log_mixup_exp=log_mixup_exp),
            RandomResizeCrop(virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)),
        )
        self.pre_norm = RunningNorm(epoch_samples=epoch_samples)
        print('Augmentatoions:', self.train_transform)

    def __call__(self, x):
        x = self.pre_norm(x)
        return self.train_transform(x), self.train_transform(x)
