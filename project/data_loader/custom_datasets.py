from typing import Any, Callable, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset
import torchaudio
from torchvision.datasets.folder import ImageFolder

from preprocessing.transforms import pipelines

SND_EXTENSIONS = ".wav"

def is_sound_file(item: str) -> bool:
    return Path(item).suffix in SND_EXTENSIONS and (not item.endswith("gitignore"))


def get_dataset(dataset_choice, data_dir):
    if dataset_choice == 'original':
        dataset = MySoundFolder(
            data_dir,
            loader=torchaudio.load,
            transform=pipelines("split", length=52, n_mels=56),
        )
    elif dataset_choice == 'custom':
        dataset = ImageFolder(data_dir,
                                   transform=pipelines("overlapping_from_image", length=52, n_mels=56),
                                   )
    return dataset


class MySoundFolder(ImageFolder):
    """
    class interfacing with audio files and doing on the fly data augmentation adding a 90 db and substracting 90 db
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:

        is_valid_file = is_valid_file or is_sound_file

        super(MySoundFolder, self).__init__(
            root, transform, target_transform, loader, is_valid_file
        )

        self.imgs = self.samples

    def __len__(self) -> int:
        return 3 * len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any,str, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, source path, case) where target is class_index of the target class.
        """
        from torchaudio.sox_effects import apply_effects_tensor

        _index = index // 3
        _case = divmod(index, 3)[1]
        path, target = self.samples[_index]
        sample = self.loader(path)
        metadata = torchaudio.info(path)
        sample_rate = metadata.sample_rate
        # channels (Optional[int]) – The number of channels
        # rate (Optional[float]) – Sampling rate
        # precision (Optional[int]) – Bit depth
        # length (Opti For sox backend, the number of samples. (frames * channels).
        # For soundfile backend, the number of frames.

        if _case == 1:
            effects = [["pitch", "90"]]
            sample = apply_effects_tensor(
                sample[0], effects=effects, sample_rate=int(sample_rate)
            )
        if _case == 2:
            effects = [["pitch", "-90"]]
            sample = apply_effects_tensor(
                sample[0], effects=effects, sample_rate=int(sample_rate)
            )
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path, _case  # sample (tensor, sample rate), target, source path, augmentation case



class PatchDatasetFromImageFolder(TensorDataset):

    def __init__(self,
                 root: str,
                 transform = pipelines("min_overlapping", length=102, n_mels=56), ) -> None:

        super(TensorDataset, self).__init__()

        dataset = ImageFolder(root, transform=transform)

        nb_patches_per_im = [len(x[0]) for x in dataset]
        patches_labels = sum(
            [[dataset[i][1]] * nb_patches_per_im[i] for i in range(len(dataset))], [])
        patches = torch.cat([torch.stack(x[0]) for x in dataset], axis=0)

        n_samples = len(patches_labels)
        permutation = np.random.permutation(n_samples)
        patches_labels = np.array(patches_labels)[permutation]
        patches = patches[permutation, ...]

        self.tensors = [torch.Tensor(patches),
                                     torch.IntTensor(patches_labels)]

