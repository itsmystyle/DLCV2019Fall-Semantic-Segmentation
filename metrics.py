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
    def __init__(self, num_classes):
        self.name = "Mean IOU"
        self.num_classes = num_classes
        self.n_correct = 0.0
        self.n = 0

    def reset(self):
        self.n_correct = 0.0
        self.n = 0

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
        self.n_correct += mean_iou
        self.n += 1

    def get_score(self):
        return self.n_correct / (self.n + 1e-20)

    def print_score(self):
        score = self.get_score()
        return "{:.5f}".format(score)
