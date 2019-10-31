import os
import random

import torch
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter

import data
import parser
from metrics import MeanIOUScore
from modules.trainer import Trainer
from modules.utils import prepare_model


if __name__ == "__main__":

    args = parser.arg_parse()

    """ create directory to save trained model and other info """
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    """ setup GPU """
    torch.cuda.set_device(args.gpu)

    """ setup random seed """
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    """ load dataset and prepare data loader """
    print("===> prepare dataloader ...")
    train_loader = torch.utils.data.DataLoader(
        data.SegData(os.path.join(args.data_dir, "train"), "train", args.augmentation),
        batch_size=args.train_batch,
        num_workers=args.workers,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        data.SegData(os.path.join(args.data_dir, "val"), "valid"),
        batch_size=args.train_batch,
        num_workers=args.workers,
        shuffle=False,
    )

    """ load model """
    model = prepare_model(args)
    model.cuda()

    """ define loss """
    criterion = nn.CrossEntropyLoss()

    """ setup metrics """
    metric = MeanIOUScore(9)

    """ setup optimizer """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    """ setup tensorboard """
    writer = SummaryWriter(os.path.join(args.save_dir, "train_info"))

    """ setup trainer """
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        args.accumulate_gradient,
        train_loader,
        val_loader,
        writer,
        metric,
        args.save_dir,
    )

    trainer.fit(args.epochs)
