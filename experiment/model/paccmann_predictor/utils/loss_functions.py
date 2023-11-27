"""Loss function definitions for paccmann."""
import torch
import torch.nn as nn


def pearsonr(x, y):
    """Compute Pearson correlation.

    Args:
        x (torch.Tensor): 1D vector
        y (torch.Tensor): 1D vector of the same size as y.

    Raises:
        TypeError: not torch.Tensors.
        ValueError: not same shape or at least length 2.

    Returns:
        Pearson correlation coefficient.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError('Function expects torch Tensors.')

    if len(x.shape) > 1 or len(y.shape) > 1:
        raise ValueError(' x and y must be 1D Tensors.')

    if len(x) != len(y):
        raise ValueError('x and y must have the same length.')

    if len(x) < 2:
        raise ValueError('x and y must have length at least 2.')

    # If an input is constant, the correlation coefficient is not defined.
    if bool((x == x[0]).all()) or bool((y == y[0]).all()):
        raise ValueError('Constant input, r is not defined.')

    mx = x - torch.mean(x)
    my = y - torch.mean(y)
    cost = (
        torch.sum(mx * my) /
        (torch.sqrt(torch.sum(mx**2)) * torch.sqrt(torch.sum(my**2)))
    )
    return torch.clamp(cost, min=-1.0, max=1.0)


def correlation_coefficient_loss(labels, predictions):
    """Compute loss based on Pearson correlation.

    Args:
        labels (torch.Tensor): reference values
        predictions (torch.Tensor): predicted values

    Returns:
        torch.Tensor: A loss that when minimized forces high squared correlation coefficient:
        \$1 - r(labels, predictions)^2\$  # noqa
    """
    return 1 - pearsonr(labels, predictions)**2


def mse_cc_loss(labels, predictions):
    """Compute loss based on MSE and Pearson correlation.

    The main assumption is that MSE lies in [0,1] range, i.e.: range is
    comparable with Pearson correlation-based loss.

    Args:
        labels (torch.Tensor): reference values
        predictions (torch.Tensor): predicted values

    Returns:
        torch.Tensor: A loss that computes the following:
        \$mse(labels, predictions) + 1 - r(labels, predictions)^2\$  # noqa
    """
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(predictions, labels)
    cc_loss = correlation_coefficient_loss(labels, predictions)
    return mse_loss + cc_loss

def r2_score(y_true, y_pred):
    """Compute R2 score.

    Args:
        y_true (torch.Tensor): reference values
        y_pred (torch.Tensor): predicted values

    Returns:
        torch.Tensor: R2 score
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2