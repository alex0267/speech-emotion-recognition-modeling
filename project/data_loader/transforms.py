import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import torchaudio
import torch
from torchvision.transforms import Lambda
from functools import partial
from torchvision.transforms import Compose

def split_into_patches(sample: tuple, length: float):
    """

    :param sample:
    :param length:
    :return:
    """
    spectrogram = sample[0]
    sample_rate = sample[1]
    patch_list = np.array_split(spectrogram[:, 0:(spectrogram.shape[1] - (spectrogram.shape[1] % length))],
                                spectrogram.shape[1] // length, axis=1)
    out = [(patch, sample_rate) for patch in patch_list]
    return out

def overlapping_patches(sample: tuple, length: float, n_mels: int = 56):
    """
    :param sample:
    :param length:
    :param n_mels:
    :return:
    """
    spectrogram = sample[0]
    sample_rate = sample[1]
    patch_list = extract_patches_2d(spectrogram, (n_mels, length))
    out = [(patch, sample_rate) for patch in patch_list]
    return out

PIPELINES = {"split": lambda length: Compose([partial(split_into_patches, length=length),
                    Lambda(lambda patchs: (torch.stack([patch[0] for patch in patchs]), patchs[0][1]))]),
             "overlapping": lambda length: torchaudio.Compose([partial(overlapping_patches, length=length),
                                                         Lambda(lambda patchs: (torch.stack([patch[0] for patch in patchs]), patchs[0][1]))])}
