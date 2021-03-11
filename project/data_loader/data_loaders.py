from base import BaseDataLoader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets.folder import ImageFolder as SoundFolder
from pathlib import Path
import torchaudio
from torchvision import transforms, datasets
from data_loader.transforms import PIPELINES
import torch

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

SND_EXTENSIONS = ('.wav')

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
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.3, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = MySoundFolder(self.data_dir,loader=torchaudio.load,transform=PIPELINES["split"](52,56))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,collate_fn=collate_fn)


class MySoundFolder(SoundFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any]=None,
            is_valid_file: Optional[Callable[[str], bool]] = lambda item: Path(item).suffix in SND_EXTENSIONS,
    ) -> None:
        super(MySoundFolder, self).__init__(root, transform,target_transform,loader,is_valid_file)
        self.imgs = self.samples

    def __len__(self) -> int:
        return 3*len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            from torchaudio.sox_effects import apply_effects_tensor

            _index = index//3
            _case = divmod(index,3)[1]
            path, target = self.samples[_index]
            sample = self.loader(path)
            info = torchaudio.info(path)[0]
            #channels (Optional[int]) – The number of channels
            #rate (Optional[float]) – Sampleing rate
            #precision (Optional[int]) – Bit depth
            #length (Opti For sox backend, the number of samples. (frames * channels). For soundfile backend, the number of frames.

            if _case==1:
                effects = [ ['pitch', '90']]
                sample = apply_effects_tensor(sample[0],effects=effects,sample_rate=int(info.rate))
            if _case==2:
                effects = [ ['pitch', '-90']]
                sample = apply_effects_tensor(sample[0],effects=effects,sample_rate=int(info.rate))
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target #sample (tensor, sample rate), target

