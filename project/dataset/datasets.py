import random
import re
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torchaudio
from PIL import Image
from torchvision.datasets.folder import ImageFolder as SoundFolder

from preprocessing.transforms import pipelines

SND_EXTENSIONS = ".wav"

import logging
logger = logging.getLogger(__name__)




def is_sound_file(item: str) -> bool:
    return Path(item).suffix in SND_EXTENSIONS and (not item.endswith("gitignore"))


def is_pic_file(item: str) -> bool:
    return Path(item).suffix in ".jpg" and (not item.endswith("gitignore"))


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
            effects = [['gain', '-n', '10']]
            sample = apply_effects_tensor(
                sample[0], effects=effects, sample_rate=int(sample_rate)
            )
        if _case == 2:
            effects = [['gain', '-n', '-10']]
            sample = apply_effects_tensor(
                sample[0], effects=effects, sample_rate=int(sample_rate)
            )
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path, _case  # sample (tensor, sample rate), target, source path, augmentation case


class PatchFolder(SoundFolder):
    """
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        loader = loader or Image.open
        is_valid_file = is_valid_file or is_pic_file
        if not transform:
            transform = pipelines("min_overlapping", length=102, n_mels=56)

        super(SoundFolder, self).__init__(
            root=root, transform=transform, target_transform=target_transform, loader=loader,
            is_valid_file=is_valid_file
        )
        self._post_init()

    def _post_init(self):
        _samples = []

        for item in self.samples:
            filepath = item[0]
            label = item[1]
            img_as_txt = filepath
            img_as_pil = self.loader(img_as_txt)
            print(filepath)
            try:
                img_as_tensor = self.transform()(img_as_pil)
                for patch in img_as_tensor:
                    _samples.append({"filepath": filepath, "label": label, "tensor": patch})
            except Exception as e:
                logger.info(f"error transforming file {filepath}")

        random.shuffle(_samples)
        self._samples = _samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any, dict]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, source path, case) where target is class_index of the target class.
        """

        rich_sample = self._samples[index]
        sample = rich_sample["tensor"]
        label = rich_sample["label"]
        name = Path(rich_sample["filepath"]).name
        rich_sample["name"] = name
        uuid_re = r".*_([^_]*).jpg"
        case_re = r"(\d)_.*.jpg"
        rich_sample["uuid"] = re.match(uuid_re, name).groups()[0]
        rich_sample["case"] = re.match(case_re, name).groups()[0]
        rich_sample["index"] = index

        return sample, label, rich_sample
