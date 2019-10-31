import numpy as np


class Metrics:
    def __init__(self):
        self.name = "Metric Name"

    def reset(self):
        pass

    def update(self, predicts, targets):
        pass

    def get_score(self):
        pass


class MeanIOUScore(Metrics):
    def __init__(self, num_classes, window_size=100):
        self.name = "Mean IOU"
        self.num_classes = num_classes
        self.window_size = window_size
        self.scores = []

    def reset(self):
        self.scores = []

    def update(self, predicts, targets):
        mean_iou = 0.0
        for i in range(self.num_classes):
            _preds = predicts == i
            _tars = targets == i
            tp_fp = _preds.sum()
            tp_fn = _tars.sum()
            tp = (_preds * _tars).sum()
            iou = tp / (tp_fp + tp_fn - tp + 1e-20)
            mean_iou += iou / self.num_classes
        self.scores.append(mean_iou)
        ws = self.window_size
        self.scores = self.scores[-ws:]

    def get_score(self):
        return np.mean(self.scores)

    def print_score(self):
        score = self.get_score()
        return "{:.5f}".format(score)
