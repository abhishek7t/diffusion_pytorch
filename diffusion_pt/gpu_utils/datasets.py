"""
Dataset loading utilities.

All images are scaled to [0, 255] instead of [0, 1]
"""

# import torch
# from torchvision import datasets
# from torchvision.datasets import CelebA, CIFAR10

# def pack(image, label: torch.Tensor):
#     label = label.type(torch.int32)
#     return {'image': image, 'label': label}


# class SimpleDataset:
#     DATASET_NAMES = ('cifar10', 'celebahq256')

#     def __init__(self, name, data_dir, split='train'):
#         self._name = name
#         self._data_dir = data_dir
#         self._split = split
#         self._img_size = {'cifar10': 32, 'celebahq256': 256}[name]
#         self._img_shape = [self._img_size, self._img_size, 3]
#         self._torchds_name = {
#         'cifar10': 'cifar10:3.0.0',
#         'celebahq256': 'celeb_a_hq/256:2.0.0',
#         }[name]
#         self.num_train_examples, self.num_eval_examples = {
#         'cifar10': (50000, 10000),
#         'celebahq256': (30000, 0),
#         }[name]
#         self.num_classes = 1  # unconditional
#         self.eval_split_name = {
#         'cifar10': 'test',
#         'celebahq256': None,
#         }[name]

#         self.datast = self._get_dataset()

#     def image_shape(self):
#         """Returns a tuple with the image shape."""
#         return tuple(self._img_shape)
    
#     def _get_dataset(self):
#         if self._name == 'cifar10':
#             return CIFAR10(root=self._data_dir, split=self._split, download=True)
#         elif self._name == 'celebahq256':
#             """
#             https://www.tensorflow.org/datasets/catalog/celeb_a_hq#celeb_a_hq256
#             https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training
#             FeaturesDict({
#                 'image': Image(shape=(256, 256, 3), dtype=uint8),
#                 'image/filename': Text(shape=(), dtype=string),
#             })
#             https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
#             """
#             return CelebA(root=self._data_dir, split=self._split, download=True)
        
        
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image

def pack(image, label):
    """Helper function to pack the image and label."""
    label = torch.tensor(label, dtype=torch.int32)
    return {'image': image, 'label': label}

class SimpleDataset(Dataset):
    DATASET_NAMES = ('cifar10', 'celebahq256')

    def __init__(self, name, data_dir, transform=None):
        """
        Custom dataset for loading CIFAR-10 or CelebA-HQ256 images.
        :param name: Dataset name ('cifar10' or 'celebahq256').
        :param data_dir: Directory where the dataset is stored.
        :param transform: Optional transformations to apply to images.
        """
        self._name = name
        self._data_dir = data_dir
        self._img_size = {'cifar10': 32, 'celebahq256': 256}[name]
        self._transform = transform or self.default_transforms()

        if name == 'cifar10':
            # Load CIFAR-10 using torchvision's built-in function
            self.dataset = CIFAR10(root=data_dir, train=True, download=True, transform=self._transform)
        elif name == 'celebahq256':
            # Load images from the local directory for CelebA-HQ256
            self.dataset = self.load_celebahq_images()
        else:
            raise ValueError(f"Dataset {name} is not supported.")

    def default_transforms(self):
        """Default transformations for resizing images and converting to Tensor."""
        return transforms.Compose([
            transforms.Resize((self._img_size, self._img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)  # Keep values in [0, 255] range like in the original TensorFlow code
        ])

    def load_celebahq_images(self):
        """Load the images from the directory for CelebA-HQ256."""
        img_files = sorted([f for f in os.listdir(self._data_dir) if f.endswith('.jpg') or f.endswith('.png')])
        return img_files

    def __len__(self):
        if self._name == 'cifar10':
            return len(self.dataset)
        elif self._name == 'celebahq256':
            return len(self.dataset)

    def __getitem__(self, idx):
        if self._name == 'cifar10':
            image, label = self.dataset[idx]
        elif self._name == 'celebahq256':
            img_path = os.path.join(self._data_dir, self.dataset[idx])
            image = Image.open(img_path).convert('RGB')  # Ensure RGB image
            image = self._transform(image)
            label = torch.tensor(0, dtype=torch.int32)  # CelebA-HQ is unconditional, so label is always 0
        return pack(image=image, label=label)

# Function to create DataLoader
def get_dataloader(name, data_dir, batch_size, num_workers=4, shuffle=True):
    dataset = SimpleDataset(name=name, data_dir=data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

# Example to use the DataLoader for CelebA-HQ 256x256 or CIFAR-10
if __name__ == "__main__":
    # Use for CelebA-HQ 256x256 dataset
    img_dir = './celebahq256/'  # Path to CelebA-HQ images
    batch_size = 64
    celeba_loader = get_dataloader(name='celebahq256', data_dir=img_dir, batch_size=batch_size)

    # Iterate over CelebA-HQ dataset
    for batch in celeba_loader:
        images = batch['image']
        labels = batch['label']
        print(images.shape, labels)

    # Use for CIFAR-10 dataset
    cifar_dir = './data/'  # Path to store/download CIFAR-10
    cifar_loader = get_dataloader(name='cifar10', data_dir=cifar_dir, batch_size=batch_size)

    # Iterate over CIFAR-10 dataset
    for batch in cifar_loader:
        images = batch['image']
        labels = batch['label']
        print(images.shape, labels)
