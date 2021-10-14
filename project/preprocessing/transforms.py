import logging

import numpy as np
import torch
import torchaudio.transforms
from sklearn.feature_extraction.image import extract_patches_2d
from torchaudio.transforms import MelSpectrogram
from torchvision import transforms
from torchvision.transforms import Compose, Lambda


class TrimSilent(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample: tuple):
        """
        Vad trims only from front, signal needs to be flipped.
        """
        waveform = sample[0]
        sample_rate = sample[1]
        # Trim the "end" of the track
        waveform = torchaudio.transforms.Vad(sample_rate)(torch.flip(waveform, [1, 0]))
        # Trim the start of the track
        return (
            torchaudio.transforms.Vad(sample_rate)(torch.flip(waveform, [1, 0])),
            sample_rate,
        )


class ToMelSpectogram(torch.nn.Module):
    """
    class to convert a sound file represented as a tuple of waveform and a sample rate a to a mel spectrogram
    """

    def __init__(self, n_mels: int):
        super().__init__()
        self.n_mels = n_mels

    def forward(self, sample: tuple):  # -> channel*#n_mels*dim
        waveform = sample[0]
        sample_rate = sample[1]
        return MelSpectrogram(sample_rate=sample_rate, n_mels=self.n_mels)(waveform)

class SplitIntoPatches(torch.nn.Module):
    """
    split an image represented by a tensor
    """

    def __init__(self, length: float):
        super().__init__()
        self.length = length

    def forward(self, sample: tuple):
        """
        :param sample:
        :return:
        """
        spectrogram = sample  # channel*#n_mels*dim
        print(sample.shape)
        patch_list = np.array_split(
            spectrogram[
            :, :, 0: (spectrogram.shape[2] - (spectrogram.shape[2] % self.length))
            ],
            spectrogram.shape[2] // self.length,
            axis=2,
        )
        return patch_list  # List[#chanel*n_mels*length]


class OverlappingPatches(torch.nn.Module):
    """
    split an image represented by a tensor (overlapping images)
    """

    def __init__(self, length: float, n_mels: int):
        super().__init__()
        self.length = length
        self.n_mels = n_mels or 56

    def forward(self, sample):
        """
        :param sample: tensor
        :return:
        """
        spectrogram = sample
        num_channels = spectrogram.shape[0]
        if num_channels == 1:
            spectrogram = spectrogram[0]

        patch_list = extract_patches_2d(spectrogram, (self.length, self.n_mels))  # image, (patch_height, patch_width)
        return patch_list


def stack_patches(patchs):
    """

    :param patchs:
    :return:
    """
    output = torch.stack([torch.from_numpy(patch) for patch in patchs])
    output = output.unsqueeze(1)
    return output


def pipelines(name, length: float, n_mels: int):
    try:
        if name == "split":
            return Compose(
                [
                    TrimSilent(),
                    ToMelSpectogram(n_mels),
                    transforms.Normalize((4.5897555,), (16.177462,)),
                    SplitIntoPatches(length),
                    Lambda(stack_patches),
                ]
            )
        if name == "overlapping":
            return Compose(
                [
                    TrimSilent(),
                    ToMelSpectogram(n_mels),
                    OverlappingPatches(length, n_mels),
                    Lambda(stack_patches),
                ]
            )
        if name == "overlapping_from_image":
            return Compose(

                [
                    transforms.ToTensor(),
                    transforms.Grayscale(num_output_channels=1),
                    OverlappingPatches(length, n_mels),
                    Lambda(stack_patches),
                ]
            )
        if name == "min_overlapping":
            return Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),SplitIntoPatches(length)])
        #

    except ValueError:
        logging.info("This pipeline is not defined")
        raise
