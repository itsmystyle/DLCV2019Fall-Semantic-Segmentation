import os

import torch
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter

import data
import parser
from modules.baseline_model import BaselineNet

# from test import evaluate


def evaluate(model, val_loader):
    pass


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":

    args = parser.arg_parse()

    """create directory to save trained model and other info"""
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    """ setup GPU """
    torch.cuda.set_device(args.gpu)

    """ setup random seed """
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    """ load dataset and prepare data loader """
    print("===> prepare dataloader ...")
    train_loader = torch.utils.data.DataLoader(
        data.SegData(os.path.join(args.data_dir, "train"), "train"),
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
    print("===> prepare model ...")
    model = BaselineNet(args)
    model.cuda()

    """ define loss """
    criterion = nn.CrossEntropyLoss()

    """ setup optimizer """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    """ setup tensorboard """
    writer = SummaryWriter(os.path.join(args.save_dir, "train_info"))

    """ train model """
    print("===> start training ...")
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch + 1):

        model.train()

        for idx, (imgs, segs) in enumerate(train_loader):

            train_info = "Epoch: [{0}][{1}/{2}]".format(
                epoch, idx + 1, len(train_loader)
            )
            iters += 1

            """ move data to gpu """
            imgs, segs = imgs.cuda(), segs.cuda()

            """ forward path """
            output = model(imgs)

            """ compute loss, backpropagation, update parameters """
            loss = criterion(output, segs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """ write out information to tensorboard """
            writer.add_scalar("loss", loss.data.cpu().numpy(), iters)
            train_info += " loss: {:.4f}".format(loss.data.cpu().numpy())

            print(train_info)

        # if epoch % args.val_epoch == 0:
        #     """ evaluate the model """
        #     acc = evaluate(model, val_loader)
        #     writer.add_scalar("val_acc", acc, iters)
        #     print("Epoch: [{}] ACC:{}".format(epoch, acc))

        #     """ save best model """
        #     if acc > best_acc:
        #         save_model(model, os.path.join(args.save_dir, "model_best.pth.tar"))
        #         best_acc = acc

        """ save model """
        save_model(model, os.path.join(args.save_dir, "model_{}.pth.tar".format(epoch)))
