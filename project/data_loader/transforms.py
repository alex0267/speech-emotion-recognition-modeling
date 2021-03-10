import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import torch
from torchvision.transforms import Lambda
from functools import partial
from torchvision.transforms import Compose
from torchaudio.transforms import MelSpectrogram


def to_mel_spectrogram(sample: tuple, n_mels: int):  # -> channel*#n_mels*dim
    waveform = sample[0]
    sample_rate = sample[1]
    return MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)


# def split_into_patches(sample: tuple, length: float):
#     """
#
#     :param sample:
#     :param length:
#     :return:
#     """
#     spectrogram = sample #channel*#n_mels*dim
#     num_channels = spectrogram.shape[0]
#     if num_channels == 1:
#         spectrogram = spectrogram[0]
#     patch_list = np.array_split(spectrogram[:, 0:(spectrogram.shape[1] - (spectrogram.shape[1] % length))],
#                                 spectrogram.shape[1] // length, axis=1)
#     return patch_list
#
def split_into_patches(sample: tuple, length: float):
    """

    :param sample:
    :param length:
    :return:
    """
    spectrogram = sample #channel*#n_mels*dim
    patch_list = np.array_split(spectrogram[:,:, 0:(spectrogram.shape[2] - (spectrogram.shape[2] % length))],
                                spectrogram.shape[2] // length, axis=2)
    return patch_list #List[#chanel*n_mels*length]

def overlapping_patches(sample: tuple, length: float, n_mels: int = 56):
    """
    :param sample:
    :param length:
    :param n_mels:
    :return:
    """
    spectrogram = sample
    num_channels = spectrogram.shape[0]
    if num_channels == 1:
        spectrogram = spectrogram[0]
    patch_list = extract_patches_2d(spectrogram, (n_mels, length))
    return patch_list


PIPELINES = {"split": lambda length, n_mels: Compose([
                 partial(to_mel_spectrogram, n_mels=n_mels),
                 partial(split_into_patches, length=length),
                 Lambda(lambda patchs: torch.stack([patch for patch in patchs]))]),
             "overlapping": lambda length, n_mels: Compose([
                 partial(to_mel_spectrogram, n_mels=n_mels),
                 partial(overlapping_patches, length=length),
                 Lambda(lambda patchs: torch.stack([patch[0] for patch in patchs]))
             ])}