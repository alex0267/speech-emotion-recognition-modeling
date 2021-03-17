from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder as SoundFolder

from base import BaseDataLoader
from data_loader.transforms import PIPELINES


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        trsfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


SND_EXTENSIONS = ".wav"


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


class DnnDataLoader(BaseDataLoader):
    """
    Dnn data loading using BaseDataLoader

    """

    def __init__(
        self, data_dir, batch_size, validation_split=0.3, num_workers=1, training=True
    ):
        self.data_dir = data_dir
        self.dataset = MySoundFolder(
            self.data_dir, loader=torchaudio.load, transform=PIPELINES["split"](52, 56)
        )

        super().__init__(
            self.dataset,
            batch_size,
            validation_split,
            num_workers,
            collate_fn=collate_fn,
        )

    def _get_class_weights(self):
        unique, counts = np.unique(
            [class_index for _, class_index in self.dataset], return_counts=True
        )
        emotion_count = dict(zip(unique, counts))

        total_count = sum(counts)

        emotion_weights_dict = dict()
        for k, v in emotion_count.items():
            emotion_weights_dict[k] = total_count / v
        return emotion_weights_dict

    def _split_sampler(self, split):
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
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        # train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_data = np.take(self.dataset, train_idx)
        emotion_dict = self._get_class_weights()
        emotion_weights = [
            emotion_dict[class_index] for _, class_index in self.dataset.imgs
        ]
        train_sampler = WeightedRandomSampler(emotion_weights, len(train_idx))

        # import ipdb; ipdb.set_trace()

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler


class MySoundFolder(SoundFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = None,
        is_valid_file: Optional[Callable[[str], bool]] = lambda item: (
            Path(item).suffix in SND_EXTENSIONS
        )
        and (not item.endswith("gitignore")),
    ) -> None:
        super(MySoundFolder, self).__init__(
            root, transform, target_transform, loader, is_valid_file
        )
        self.imgs = self.samples

    def __len__(self) -> int:
        return 3 * len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        from torchaudio.sox_effects import apply_effects_tensor

        _index = index // 3
        _case = divmod(index, 3)[1]
        path, target = self.samples[_index]
        sample = self.loader(path)
        info = torchaudio.info(path)[0]
        # channels (Optional[int]) – The number of channels
        # rate (Optional[float]) – Sampleing rate
        # precision (Optional[int]) – Bit depth
        # length (Opti For sox backend, the number of samples. (frames * channels). For soundfile backend, the number of frames.

        if _case == 1:
            effects = [["pitch", "90"]]
            sample = apply_effects_tensor(
                sample[0], effects=effects, sample_rate=int(info.rate)
            )
        if _case == 2:
            effects = [["pitch", "-90"]]
            sample = apply_effects_tensor(
                sample[0], effects=effects, sample_rate=int(info.rate)
            )
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target  # sample (tensor, sample rate), target
