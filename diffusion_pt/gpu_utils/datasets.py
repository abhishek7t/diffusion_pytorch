"""
Dataset loading utilities.

All images are scaled to [0, 255] instead of [0, 1]
"""

import torch
from torchvision import datasets

def pack(image, label: torch.Tensor):
    label = label.type(torch.int32)
    return {'image': image, 'label': label}


class SimpleDataset:
    DATASET_NAMES = ('cifar10', 'celebahq256')

    def __init__(self, name, data_dir, train=True):
        self._name = name
        self._data_dir = data_dir
        self._train = train
        self._img_size = {'cifar10': 32, 'celebahq256': 256}[name]
        self._img_shape = [self._img_size, self._img_size, 3]
        self._torchds_name = {
        'cifar10': 'cifar10:3.0.0',
        'celebahq256': 'celeb_a_hq/256:2.0.0',
        }[name]
        self.num_train_examples, self.num_eval_examples = {
        'cifar10': (50000, 10000),
        'celebahq256': (30000, 0),
        }[name]
        self.num_classes = 1  # unconditional
        self.eval_split_name = {
        'cifar10': 'test',
        'celebahq256': None,
        }[name]

        self.datast = self._get_dataset()

    def image_shape(self):
        """Returns a tuple with the image shape."""
        return tuple(self._img_shape)
    
    def _get_dataset(self):
        if self._name == 'cifar10':
            return datasets.CIFAR10(root=self._data_dir, train=self._train, download=True)