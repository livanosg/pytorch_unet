import torch
from torch.nn import Module
import torch.nn.functional as F
eps = torch.tensor(1e-20)


class WeightedCrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, onehot_labels):
        num_classes = output.shape[1]
        class_frequencies = torch.sum(onehot_labels, dim=(0, 2, 3))
        # Data balancing
        weights = torch.div(torch.sum(class_frequencies), torch.add(torch.mul(class_frequencies, num_classes),eps))
        weights = weights.view([-1, 2, 1, 1])
        weight_map = torch.mul(onehot_labels, weights)
        weight_map = torch.sum(weight_map, dim=1, keepdim=True)
        loss = F.binary_cross_entropy(output, onehot_labels, weight=weight_map)
        return loss


class WeightedDiceLoss(Module):
    """Weighted Dice Loss computed from class-map label."""

    def __init__(self):
        super().__init__()

    def forward(self, output, onehot_label):
        class_frequencies = torch.sum(onehot_label, dim=(0, 2, 3))
        weights = torch.div(1., torch.pow(class_frequencies, 2) + eps)
        weights.requires_grad = False
        weights = weights.to(output.device)
        numerator = torch.sum(output * onehot_label, dim=(0, 2, 3))
        numerator = 2.0 * torch.sum(weights * numerator)
        denominator = torch.sum(onehot_label + output, dim=(0, 2, 3))
        denominator = torch.sum(weights * denominator)
        loss = -torch.log(numerator / denominator)
        return loss


def custom_loss(output, onehot_label):
    wdc = WeightedDiceLoss()
    wce = WeightedCrossEntropyLoss()
    return wdc(output, onehot_label) * .2 + wce(output, onehot_label) * .8
