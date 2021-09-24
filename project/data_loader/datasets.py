import re
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torchvision.datasets.folder import ImageFolder as SoundFolder

from data_loader.transforms import pipelines

SND_EXTENSIONS = ".wav"


def is_sound_file(item: str) -> bool:
    return Path(item).suffix in SND_EXTENSIONS and (not item.endswith("gitignore"))


class MySoundFolder(SoundFolder):
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

    def __getitem__(self, index: int) -> Tuple[Any, Any, str, int]:
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


class MyPatchFolder(SoundFolder):
    """
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = pipelines("min_overlapping", length=52, n_mels=56),
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(SoundFolder, self).__init__(
            root, transform, target_transform, loader, is_valid_file
        )
        self._post_init()

    def _post_init(self):
        nb_patches_per_im = [len(x[0]) for x in self.samples]
        patches_labels = sum([[self.samples[i][1]] * nb_patches_per_im[i] for i in range(len(self.samples))], [])
        patches = torch.cat([torch.stack(x[0]) for x in self.samples], axis=0)

        n_samples = len(patches_labels)
        permutation = np.random.permutation(n_samples)
        patches_labels = np.array(patches_labels)[permutation]
        patches = patches[permutation, ...]

        self.samples = [torch.Tensor(patches), torch.IntTensor(patches_labels)]
        self.imgs = self.samples

    def __getitem__(self, index: int) -> Tuple[Any, Any, str, str]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, source path, case) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        name = Path(path).name
        uuid_re = r".*_([^_]*).jpg"
        case_re = r"(\d)_.*.jpg"
        return sample, target, re.match(uuid_re, name).groups()[0], re.match(case_re, name).groups()[0]
