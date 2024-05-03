from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
from torchvision.transforms import ToTensor, Resize, Normalize, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter


# Fine-tuning dataset
# train_mean = [0.59685254, 0.59685254, 0.59685254]
# train_std = [0.16043035, 0.16043035, 0.16043035]

# ImageNet dist
train_mean = [0.485, 0.456, 0.406]
train_std = [0.229, 0.224, 0.225]


class CustomDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data  # Df containing filepaths and labels of imgs
        self.mode = mode  # training or test

        # Data transformations
        self._transform_t = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            # tv.transforms.RandomCrop((300, 300), 30, True, padding_mode='reflect'),
            # tv.transforms.ToPILImage(),
            # tv.transforms.Resize(299),
            tv.transforms.Resize((300,300)),
            tv.transforms.RandomHorizontalFlip(0.5),
            tv.transforms.RandomVerticalFlip(0.5),
            # tv.transforms.RandomRotation(360),
            # tv.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # tv.transforms.ColorJitter(brightness=0.4, contrast=0.4),
            # tv.transforms.ColorJitter(contrast=0.4,saturation=0.4),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std),
        ])
        self._transform_v = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.Resize((300,300)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        channel_last = imread(self.data.iloc[index]["filename"])
        # img = np.transpose(channel_last, (2, 0, 1))

        # load labels
        label = torch.tensor(self.data.iloc[index, 1:], dtype=torch.float32)

        # data transformations
        if self.mode == 'training':
            transformed = self._transform_t(channel_last)
        else:
            transformed = self._transform_v(channel_last)

        return transformed, label