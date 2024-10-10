import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image

class SimpleDataset(Dataset):
    DATASET_NAMES = ('cifar10', 'celebahq256')

    def __init__(self, name, data_dir, transform=None, img_size=None):
        """
        Custom dataset for loading CIFAR-10 or CelebA-HQ256 images.
        :param name: Dataset name ('cifar10' or 'celebahq256').
        :param data_dir: Directory where the dataset is stored.
        :param transform: Optional transformations to apply to images.
        :param img_size: Desired image size (default is 32 for CIFAR-10, 256 for CelebA-HQ256).
        """
        self._name = name
        self._data_dir = data_dir
        self._img_size = img_size or {'cifar10': 32, 'celebahq256': 256}[name]
        self._transform = transform
        if self._transform is None:
            self._transform = transforms.Compose([
                transforms.Resize((self._img_size, self._img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 2) - 1)  # Scale from [0, 1] to [-1, 1]
            ])

        if name == 'cifar10':
            # Load CIFAR-10 using torchvision's built-in function
            self.dataset = CIFAR10(root=data_dir, train=True, download=True, transform=self._transform)
        elif name == 'celebahq256':
            # Load images from the local directory for CelebA-HQ256
            self.dataset = self.load_celebahq_images()
        else:
            raise ValueError(f"Dataset {name} is not supported.")

    def load_celebahq_images(self):
        """Load the images from the directory for CelebA-HQ256."""
        if not os.path.exists(self._data_dir):
            raise FileNotFoundError(f"Directory '{self._data_dir}' does not exist.")
        img_files = sorted([
            f for f in os.listdir(self._data_dir)
            if os.path.isfile(os.path.join(self._data_dir, f)) and (f.endswith('.jpg') or f.endswith('.png'))
        ])
        if not img_files:
            raise FileNotFoundError(f"No images found in directory '{self._data_dir}'.")
        return img_files

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._name == 'cifar10':
            image, label = self.dataset[idx]
        elif self._name == 'celebahq256':
            img_path = os.path.join(self._data_dir, self.dataset[idx])
            image = Image.open(img_path).convert('RGB')  # Ensure RGB image
            image = self._transform(image)
            label = 0  # CelebA-HQ is unconditional, so label is always 0
        else:
            raise ValueError(f"Dataset {self._name} is not supported.")
        return {'image': image, 'label': torch.tensor(label, dtype=torch.long)}

# Function to create DataLoader
def get_dataloader(name, data_dir, batch_size, num_workers=4, shuffle=True, img_size=None):
    transform_list = [
        transforms.Resize((img_size or {'cifar10': 32, 'celebahq256': 256}[name],) * 2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2) - 1)  # Scale from [0, 1] to [-1, 1]
    ]
    transform = transforms.Compose(transform_list)
    dataset = SimpleDataset(name=name, data_dir=data_dir, transform=transform, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
