import numpy as np
import torch
import torchaudio
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

from .base.base_data_loader import BaseDataLoader
from data_loader.transforms import pipelines
from dataset.datasets import MySoundFolder
from utils.util import set_seed

torchaudio.set_audio_backend("sox_io")


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

from dataset.datasets import PatchFolder

class PatchDnnDataLoader(BaseDataLoader):
    """
    Dnn data loading using BaseDataLoader

    """

    def __init__(
            self,
            data_dir,
            batch_size,
            validation_split=0.3,
            num_workers=1,
            seed=0
    ):
        self.data_dir = data_dir
        self.dataset = PatchFolder(self.data_dir)
        set_seed(seed)
        super().__init__(
            self.dataset,
            batch_size,
            validation_split,
            num_workers
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

        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert (
                    split < self.n_samples
            ), "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        # WE ALWAYS TAKE THE FIRSTS ELEMENTS FOR VALIDATION?
        valid_idx = idx_full[0:len_valid]
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        #emotion_dict = self._get_class_weights()
        #emotion_weights = [emotion_dict[np.int(self.dataset[i][1])] for i in train_idx]
        emotion_weights = [1 for i in train_idx]
        train_sampler = WeightedRandomSampler(emotion_weights, len(train_idx))
        #train_sampler = SubsetRandomSampler(len(train_idx))

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler


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
            seed=0
    ):
        from torchvision.datasets.folder import ImageFolder
        set_seed(seed)
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

        assert(len(set(list(train_sampler)).intersection(list(valid_sampler)))==0)
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
        seed=0
    ):
        set_seed(seed)
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

