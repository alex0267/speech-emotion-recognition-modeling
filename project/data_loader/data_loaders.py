from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder as SoundFolder

from base import BaseDataLoader
from data_loader.transforms import pipelines

torchaudio.set_audio_backend("sox_io")


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        validation_split=0.0,
        num_workers=1,
        training=True,
        shuffle=True,
    ):
        trsfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm
        )
        super().__init__(
            self.dataset, batch_size, validation_split, num_workers, shuffle=shuffle
        )


SND_EXTENSIONS = ".wav"


def is_sound_file(item: str) -> bool:
    return Path(item).suffix in SND_EXTENSIONS and (not item.endswith("gitignore"))


def collate_fn(batch):
    """
    flatten stacked tensors
    :param batch:
    :return:
    """
    data_list, label_list = [], []
    for _data, _label in batch:
        for tsr in torch.unbind(_data):
            data_list.append(tsr)
            label_list.append(_label)
    return torch.stack(data_list).float(), torch.IntTensor(label_list)




class CustomDnnDataLoader(BaseDataLoader):
    """
    Dnn data loading using BaseDataLoader

    """

    def __init__(
            self,
            data_dir,
            batch_size,
            validation_split=0.3,
            num_workers=1,
    ):
        from torchvision.datasets.folder import ImageFolder
        self.data_dir = data_dir
        self.dataset = ImageFolder(self.data_dir,
            transform=pipelines("overlapping_from_image", length=52, n_mels=56),
        )

        super().__init__(
            self.dataset,
            batch_size,
            validation_split,
            num_workers,
            collate_fn=collate_fn,
        )

    def _get_class_weights(self):
        """
        compute a dict with weights for each class
        :return:
        """
        unique, counts = np.unique(
            [class_index for _, class_index in self.dataset], return_counts=True
        )
        emotion_count = dict(zip(unique, counts))
        total_count = sum(counts)

        return {k: total_count / v for k, v in emotion_count.items()}

    def _split_sampler(self, split):
        """
        split a sample reweighting classes using WeightedRandomSampler
        returning a training set and a validation set
        :param split: either an int giving the number of sample in the validation set or
        a float, giving the ratio of the validation set
        :return:
        """
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert (
                    split < self.n_samples
            ), "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        emotion_dict = self._get_class_weights()
        emotion_weights = [emotion_dict[self.dataset[i][1]] for i in train_idx]
        train_sampler = WeightedRandomSampler(emotion_weights, len(train_idx))

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler



class DnnDataLoader(BaseDataLoader):
    """
    Dnn data loading using BaseDataLoader

    """

    def __init__(
        self,
        data_dir,
        batch_size,
        validation_split=0.3,
        num_workers=1,
    ):
        self.data_dir = data_dir
        self.dataset = MySoundFolder(
            self.data_dir,
            loader=torchaudio.load,
            transform=pipelines("split", length=52, n_mels=56),
        )

        super().__init__(
            self.dataset,
            batch_size,
            validation_split,
            num_workers,
            collate_fn=collate_fn,
        )

    def _get_class_weights(self):
        """
        compute a dict with weights for each class
        :return:
        """
        unique, counts = np.unique(
            [class_index for _, class_index in self.dataset], return_counts=True
        )
        emotion_count = dict(zip(unique, counts))
        total_count = sum(counts)

        return {k: total_count / v for k, v in emotion_count.items()}

    def _split_sampler(self, split):
        """
        split a sample reweighting classes using WeightedRandomSampler
        returning a training set and a validation set
        :param split: either an int giving the number of sample in the validation set or
        a float, giving the ratio of the validation set
        :return:
        """
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert (
                split < self.n_samples
            ), "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        emotion_dict = self._get_class_weights()
        emotion_weights = [emotion_dict[self.dataset[i][1]] for i in train_idx]
        train_sampler = WeightedRandomSampler(emotion_weights, len(train_idx))

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler


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