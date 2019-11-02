import os
import glob
import random

import torch
import argparse
import scipy.misc
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import hflip


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SegData(Dataset):
    def __init__(self, data_dir, mode="test", augmentation=False):
        self.mode = mode
        self.data_dir = data_dir
        self.augmentation = augmentation

        """ set up data """
        if self.mode == "test":
            """ read the data list """
            imgs_path = glob.glob(os.path.join(self.data_dir, "*.png"))
            imgs_path.sort()

            self.data = imgs_path

        else:
            """ set up basic parameters for train/validation dataset """
            self.img_dir = os.path.join(self.data_dir, "img")
            self.seg_dir = os.path.join(self.data_dir, "seg")

            """ read the data list """
            imgs_path = glob.glob(os.path.join(self.img_dir, "*.png"))
            imgs_path.sort()
            segs_path = glob.glob(os.path.join(self.seg_dir, "*.png"))
            segs_path.sort()

            self.data = [p for p in zip(imgs_path, segs_path)]

        """ set up image trainsform """
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.mode == "test":
            """ get data """
            img_path = self.data[idx]

            """ read image and convert to tensor"""
            img = Image.open(img_path).convert("RGB")

            return self.transform(img), img_path.split("/")[-1]

        else:
            """ get data """
            img_path, seg_path = self.data[idx]

            """ read image/seg and convert to tensor"""
            img = Image.open(img_path).convert("RGB")
            seg = scipy.misc.imread(seg_path)

            """ data augmentation during training """
            if self.mode == "train" and self.augmentation and random.randrange(0, 2) == 1:
                img = hflip(img)
                seg = np.asarray(hflip(Image.fromarray(seg)))

            return (self.transform(img), torch.from_numpy(seg).long())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Unit test Arguments.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("hw2_data", "train"),
        help="root path to data directory",
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="Dataloader mode for train/valid/test."
    )
    parser.add_argument(
        "--augmentation",
        dest="augmentation",
        action="store_true",
        help="Whether to do data augmentation.",
    )
    args = parser.parse_args()

    seg_data = SegData(args.data_dir, args.mode, args.augmentation)

    dataloader = DataLoader(seg_data, batch_size=32, shuffle=False, num_workers=0)
    for batch in dataloader:
        import matplotlib.pyplot as plt

        for idx, (img, seg) in enumerate(zip(*batch)):
            plt.imsave("test{}_img.png".format(idx), img.cpu().detach().numpy().transpose(1, 2, 0))
            plt.imsave("test{}_seg.png".format(idx), seg.cpu().detach().numpy())

        break
