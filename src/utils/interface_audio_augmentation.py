import augment
import torch
import numpy as np


def audio_augment_baseline(x, sr):
    random_pitch_shift = lambda: np.random.randint(-300, +300)
    random_room_size = lambda: np.random.randint(0, 101)
    noise_generator = lambda: torch.zeros_like(x).uniform_()

    combination = augment.EffectChain() \
        .additive_noise(noise_generator, snr=15) \
        .pitch("-q", random_pitch_shift).rate(sr) \
        .reverb(50, 50, random_room_size).channels(1)
    y = combination.apply(x, src_info={'rate': sr}, target_info={'rate': sr})
    return y
