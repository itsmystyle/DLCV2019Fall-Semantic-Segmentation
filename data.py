import os
import glob

import torch
import argparse
import scipy.misc
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SegData(Dataset):
    def __init__(self, data_dir, mode="test"):

        """ set up basic parameters for dataset """
        self.mode = mode
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, "img")
        self.seg_dir = os.path.join(self.data_dir, "seg")

        """ read the data list """
        imgs_path = glob.glob(os.path.join(self.img_dir, "*.png"))
        imgs_path.sort()
        segs_path = glob.glob(os.path.join(self.seg_dir, "*.png"))
        segs_path.sort()

        """ set up image path """
        if self.mode == "test":
            self.data = imgs_path
        else:
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

            return self.transform(img)
        else:
            """ get data """
            img_path, seg_path = self.data[idx]

            """ read image/seg and convert to tensor"""
            img = Image.open(img_path).convert("RGB")
            seg = scipy.misc.imread(seg_path)

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
        "--mode",
        type=str,
        default="train",
        help="Dataloader mode for train/valid/test.",
    )
    args = parser.parse_args()

    seg_data = SegData(args.data_dir, args.mode)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(seg_data, batch_size=32, shuffle=False, num_workers=0)
    for batch in dataloader:
        break
