import torch
from torch.nn import Module

eps = torch.tensor(1e-20)


class WeightedCrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, onehot_labels):
        """

        :param output: output with shape bchw
        :param onehot_labels: onehot labels with shape bchw
        :type onehot_labels: tensor
        """
        class_frequencies = torch.sum(onehot_labels, dim=[0, 2, 3], keepdim=True)
        # wtf is this
        # num_classes = output.shape[1]
        # weights = torch.div(torch.sum(class_frequencies), torch.add(torch.mul(class_frequencies, num_classes), eps))
        # Weighted cross-entropy normal data balancing
        weights = torch.div(torch.sum(class_frequencies) - class_frequencies, torch.add(class_frequencies, eps))
        wce = torch.sum(weights * onehot_labels * torch.log(output + eps))
        loss = torch.neg(torch.div(wce, torch.sum(class_frequencies)))
        print(loss)
        return loss


class WeightedDiceLoss(Module):
    """Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations.
    Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M. Jorge Cardoso"""

    def __init__(self):
        super().__init__()

    def forward(self, output, onehot_label):
        class_frequencies = torch.sum(onehot_label, dim=(0, 2, 3))
        weights = torch.div(1., torch.add(torch.pow(class_frequencies, 2.), eps)).to(output.device)
        numerator = torch.mul(output, onehot_label)
        numerator = torch.sum(numerator, dim=(0, 2, 3))
        numerator = torch.sum(torch.mul(numerator, weights)) + eps

        denominator = torch.sum(torch.mul(torch.sum(onehot_label + output, dim=(0, 2, 3)), weights))
        loss = torch.neg(torch.log(torch.div(2.0 * numerator, denominator)))
        print(loss)
        return loss


def custom_loss(output, onehot_label):
    wdc = WeightedDiceLoss()
    wce = WeightedCrossEntropyLoss()
    return wdc(output, onehot_label) * .3 + wce(output, onehot_label) * .7
