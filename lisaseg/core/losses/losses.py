import torch
import torch.nn.functional as F

def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, scale=1000, eps=1e-6):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape. The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = torch.sigmoid(inputs).flatten(start_dim=1, end_dim=2)
    targets = targets.flatten(start_dim=1, end_dim=2)
    
    numerator = 2 * (inputs * targets).sum(dim=-1)
    denominator = (inputs + targets).sum(dim=-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    
    return loss

def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Args:
        inputs: A float tensor of arbitrary shape. The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(start_dim=1, end_dim=2).mean(dim=1).sum() / (num_masks + 1e-8)
    return loss
