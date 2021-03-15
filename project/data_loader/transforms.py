import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import torch
from torchvision.transforms import Lambda
from functools import partial
from torchvision.transforms import Compose
from torchaudio.transforms import MelSpectrogram
from torchvision import transforms


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
                 transforms.Normalize((4.5897555,), (16.177462,)),
                 partial(split_into_patches, length=length),
                 Lambda(lambda patchs: torch.stack([patch for patch in patchs]))]),
             "overlapping": lambda length, n_mels: Compose([
                 partial(to_mel_spectrogram, n_mels=n_mels),
                 partial(overlapping_patches, length=length),
                 Lambda(lambda patchs: torch.stack([patch[0] for patch in patchs]))
             ])}

# import numpy as np
# items=MySoundFolder(root=str(Path(ROOT,"data","raw_data")),loader=torchaudio.load,transform=PIPELINES["split"](52,56))
# out = [item[0] for item in items]
# out = [torch.unbind(item) for item in out]
# out = torch.stack([subitem for subitem in item for item in out]).float().numpy()
# mn=np.mean(out, axis=(0,1,2))
# st=np.std(out, axis=(0,1,2))
