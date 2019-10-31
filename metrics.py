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
    def __init__(self, num_classes, window_size=128):
        self.name = "mIoU"
        self.num_classes = num_classes
        self.window_size = window_size
        self.tp_fp = [0.0] * self.num_classes
        self.tp_fn = [0.0] * self.num_classes
        self.tp = [0.0] * self.num_classes
        self.score = 0.0

    def reset(self):
        self.tp_fp = [0.0] * self.num_classes
        self.tp_fn = [0.0] * self.num_classes
        self.tp = [0.0] * self.num_classes
        self.score = 0.0

    def update(self, predicts, targets):
        mean_iou = 0.0
        for i in range(self.num_classes):
            _preds = predicts == i
            _tars = targets == i
            self.tp_fp[i] += _preds.sum()
            self.tp_fn[i] += _tars.sum()
            self.tp[i] += (_preds * _tars).sum()
            iou = self.tp[i] / (self.tp_fp[i] + self.tp_fn[i] - self.tp[i] + 1e-20)
            mean_iou += iou / self.num_classes
        self.score = mean_iou

    def get_score(self):
        return self.score

    def print_score(self):
        score = self.get_score()
        return "{:.5f}".format(score)
