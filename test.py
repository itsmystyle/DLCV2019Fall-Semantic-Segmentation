import os
import random

import torch
import numpy as np
import torch.nn.functional as F
from scipy.misc import toimage

import data
import parser
from modules.baseline_model import BaselineNet


if __name__ == "__main__":
    args = parser.arg_parse()

    """ check args """
    if not os.path.exists(args.resume):
        raise FileNotFoundError("Please provide which model to load, --resume <saved_model_file>")
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(
            "Please provide which dataset to inference, --data_dir <path_to_directory>"
        )
    if not os.path.exists(args.output_dir):
        raise FileNotFoundError(
            "Please provide which directory to save output, --output_dir <path_to_directory>"
        )

    """ setup GPU """
    torch.cuda.set_device(args.gpu)

    """ setup random seed """
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    """ load dataset and prepare data loader """
    print("===> prepare dataloader ...")
    data_loader = torch.utils.data.DataLoader(
        data.SegData(args.data_dir, "test"),
        batch_size=args.test_batch,
        num_workers=args.workers,
        shuffle=False,
    )

    """ TODO: use the save function to prepare model in train/test.py """
    """ load model """
    print("===> prepare model ...")
    model = BaselineNet(args)
    model.cuda()

    """ resume save model """
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    """ inference image segmentation """
    print("===> inferencing ...")
    predicts = []
    paths = []
    with torch.no_grad():
        for idx, (imgs, filenames) in enumerate(data_loader):
            imgs = imgs.cuda()

            preds = model(imgs)

            preds = F.softmax(preds, dim=1)
            preds = preds.max(dim=1)[1]

            predicts.append(preds.cpu().numpy())
            paths += [os.path.join(args.output_dir, filename) for filename in filenames]

    predicts = np.concatenate(predicts)

    for pred, path in zip(predicts, paths):
        toimage(pred, cmin=0, cmax=255).save(path)
