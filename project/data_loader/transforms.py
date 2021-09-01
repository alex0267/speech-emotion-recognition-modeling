import numpy as np
import torch
from sklearn.feature_extraction.image import extract_patches_2d
from torchaudio.transforms import MelSpectrogram
from torchvision import transforms
from torchvision.transforms import Compose, Lambda


class ToMelSpectogram(torch.nn.Module):
    def __init__(self, n_mels: int):
        super().__init__()
        self.n_mels = n_mels

    def forward(self, sample: tuple):  # -> channel*#n_mels*dim
        waveform = sample[0]
        sample_rate = sample[1]
        return MelSpectrogram(sample_rate=sample_rate, n_mels=self.n_mels)(waveform)


class SplitIntoPatches(torch.nn.Module):
    def __init__(self, length: float):
        super().__init__()
        self.length = length

    def forward(self, sample: tuple):
        """
        :param sample:
        :param length:
        :return:
        """
        spectrogram = sample  # channel*#n_mels*dim
        patch_list = np.array_split(
            spectrogram[
                :, :, 0: (spectrogram.shape[2] - (spectrogram.shape[2] % self.length))
            ],
            spectrogram.shape[2] // self.length,
            axis=2,
        )
        return patch_list  # List[#chanel*n_mels*length]


class OverlappingPatches(torch.nn.Module):
    def __init__(self, length: float, n_mels: int):
        super().__init__()
        self.length = length
        self.n_mels = n_mels or 56

    def forward(self, sample: tuple):
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
        patch_list = extract_patches_2d(spectrogram, (self.n_mels, self.length))
        return patch_list


def stack_patches(patchs):
    return torch.stack([patch for patch in patchs])


def pipelines(name, length: float, n_mels: int):
    try:
        if name == "split":
            return Compose(
                [
                    ToMelSpectogram(n_mels),
                    transforms.Normalize((4.5897555,), (16.177462,)),
                    SplitIntoPatches(length),
                    Lambda(stack_patches),
                ]
            )
        if name == "overlapping":
            return Compose(
                [
                    ToMelSpectogram(n_mels),
                    OverlappingPatches(length),
                    Lambda(stack_patches),
                ]
            )
    except ValueError:
        print("This pipeline is not defined")
        raise
