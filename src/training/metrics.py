import torch


def pred_digits_to_values(y_pred: torch.Tensor) -> torch.Tensor:
    """function that converts the predicted digits to the binary values
    Example: -5.36 -> 0, -1.3 -> 1"""
    return torch.round(torch.sigmoid(y_pred))


def multi_label_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float | torch.Tensor:
    y_pred = pred_digits_to_values(y_pred)
    # num of hits / num of classes mean over the batch
    return torch.mean((y_pred == y_true).sum(dim=1) / y_pred.size(1))
