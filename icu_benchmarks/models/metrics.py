import torch
from typing import Callable
import numpy as np
from ignite.metrics import EpochMetric


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def balanced_accuracy_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    try:
        from sklearn.metrics import balanced_accuracy_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy()
    y_pred = np.argmax(y_preds.numpy(), axis=-1)
    return balanced_accuracy_score(y_true, y_pred)


def ece_curve_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    try:
        from sklearn.calibration import calibration_curve
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return calibration_curve(y_true, y_pred, n_bins=10)


def mae_with_invert_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor, invert_fn=Callable) -> float:
    try:
        from sklearn.metrics import mean_absolute_error
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = invert_fn(y_targets.numpy().reshape(-1, 1))[:, 0]
    y_pred = invert_fn(y_preds.numpy().reshape(-1, 1))[:, 0]
    return mean_absolute_error(y_true, y_pred)


class BalancedAccuracy(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(BalancedAccuracy, self).__init__(
            balanced_accuracy_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn
        )


class CalibrationCurve(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(CalibrationCurve, self).__init__(
            ece_curve_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn
        )


class MAE(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False,
                 invert_transform: Callable = lambda x: x) -> None:
        super(MAE, self).__init__(
            lambda x, y: mae_with_invert_compute_fn(x, y, invert_transform), output_transform=output_transform,
            check_compute_fn=check_compute_fn
        )


def plot_confusion_matrix(cm):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
    """
    try:
        cm = cm.detach().cpu().numpy()
    except:
        pass
        
    figure = plt.figure(figsize=(4, 4))
    plt.matshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, f'{cm[i, j]}', horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_curve