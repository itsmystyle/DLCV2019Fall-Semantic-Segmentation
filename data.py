import os
import glob

import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


class SegData(Dataset):
    def __init__(self, data_dir, mode="test"):

        """ set up basic parameters for dataset """
        self.mode = mode
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, "img")
        self.seg_dir = os.path.join(self.data_dir, "seg")

        """ read the data list """
        imgs_path = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        segs_path = sorted(glob.glob(os.path.join(self.seg_dir, "*.png")))

        """ set up image path """
        if self.mode == "test":
            self.data = imgs_path
        else:
            self.data = [p for p in zip(imgs_path, segs_path)]

        """ set up image trainsform """
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.mode == "test":
            """ get data """
            img_path = self.data[idx]

            """ read image and convert to tensor"""
            img = Image.open(img_path).convert("RGB")

            return self.transform(img)
        else:
            """ get data """
            img_path, seg_path = self.data[idx]

            """ read image/seg and convert to tensor"""
            img = Image.open(img_path).convert("RGB")
            seg = Image.open(seg_path)

            return (
                self.transform(img),
                torch.from_numpy(np.array(seg, np.int32, copy=False)),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Unit test Arguments.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("hw2_data", "train"),
        help="root path to data directory",
    )
    args = parser.parse_args()

    seg_data = SegData(args.data_dir)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(seg_data, batch_size=32, shuffle=False, num_workers=4)
    for batch in dataloader:
        break
