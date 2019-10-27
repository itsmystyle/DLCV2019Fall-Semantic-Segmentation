import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from mean_iou_evaluate import mean_iou_score


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        writer,
        metric,
        save_dir,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.metric = metric
        self.save_dir = save_dir

    def fit(self, epochs):
        """ train model """
        print("===> start training ...")
        iters = -1
        val_iters = -1
        best_iou = 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_iou, iters = self._run_one_epoch(epoch, iters)

            val_loss, val_iou, val_iters, best_iou = self._eval_one_epoch(
                epoch, val_iters, best_iou
            )

            print("Train loss:{:.7f}".format(train_loss))
            print("Train Mean IOU:{:.7f}".format(train_iou))
            print("Valid loss:{:.7f}".format(val_loss))
            print("Valid Mean IOU:{:.7f}".format(val_iou))
            print()

            """ save model """
            self.save(os.path.join(self.save_dir, "model_{}.pth.tar".format(epoch)))

    def _run_one_epoch(self, epoch, iters):
        """ Run one epoch

        Returns:
        ========
            loss: float,
            mean_iou_score: float,
            iters: int
        """

        trange = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Epoch {}".format(epoch),
        )

        """ Trainining process """
        self.model.train()
        self.metric.reset()
        batch_loss = 0.0

        for idx, (imgs, segs) in trange:
            iters += 1

            """ move data to gpu """
            imgs, segs = imgs.cuda(), segs.cuda()

            """ forward path """
            preds = self.model(imgs)

            """ compute loss, backpropagation, update parameters """
            loss = self.criterion(preds, segs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            """ update metric """
            preds = F.softmax(preds, dim=1)
            preds = preds.max(dim=1)[1]
            self.metric.update(preds.data.cpu().numpy(), segs.data.cpu().numpy())

            """ update loss """
            batch_loss += loss.item()

            """ write out information to tensorboard """
            self.writer.add_scalar("train_loss", loss.data.cpu().item(), iters)
            self.writer.add_scalar("train_iou", self.metric.get_score(), iters)

            """ print loss and metrics """
            trange.set_postfix(
                loss=batch_loss / (idx + 1),
                **{self.metric.name: self.metric.print_score()}
            )

        return batch_loss / (idx + 1), self.metric.get_score(), iters

    def _eval_one_epoch(self, epoch, iters, best_iou):
        """ Evaluate one epoch

        Returns:
        ========
            loss: float
            mean_iou_score: float
            iters: int
            best_iou: float
        """

        """ Evaluating Process """
        self.model.eval()
        self.metric.reset()
        batch_loss = 0.0

        _preds_tmp = []
        _segs_tmp = []

        """ evaluate the model """
        with torch.no_grad():
            for idx, (imgs, segs) in enumerate(self.val_loader):
                iters += 1

                """ move data to gpu """
                imgs, segs = imgs.cuda(), segs.cuda()

                """ forward path """
                preds = self.model(imgs)

                """ compute loss """
                loss = self.criterion(preds, segs)

                """ update metric """
                preds = F.softmax(preds, dim=1)
                preds = preds.max(dim=1)[1]
                # self.metric.update(preds.cpu().numpy(), segs.cpu().numpy())
                _preds_tmp.append(preds.cpu().numpy())
                _segs_tmp.append(segs.cpu().numpy())

                """ update loss """
                batch_loss += loss.item()

                """ write out information to tensorboard """
                self.writer.add_scalar("val_loss", loss.data.cpu().numpy(), iters)
                # self.writer.add_scalar("val_iou", self.metric.get_score(), iters)

            # val_iou = self.metric.get_score()
            xx = np.concatenate(_preds_tmp)
            yy = np.concatenate(_segs_tmp)
            val_iou = mean_iou_score(xx, yy)

            """ save best model """
            if val_iou > best_iou:
                print("Best model saved!")
                self.save(os.path.join(self.save_dir, "model_best.pth.tar"))
                best_iou = val_iou

        return batch_loss / (idx + 1), val_iou, iters, best_iou

    def save(self, path):
        torch.save(self.model.state_dict(), path)
