import torch
from torch.nn import Module
import torch.nn.functional as F
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
        weights = torch.div(torch.sum(class_frequencies)-class_frequencies, torch.add(class_frequencies, 1))
        wce = torch.sum(weights * onehot_labels * torch.log(output + eps), dim=[0, 2, 3], keepdim=True)
        wce = - torch.sum(wce)/torch.sum(class_frequencies)
        print(wce)
        # loss = F.binary_cross_entropy(output, onehot_labels) #, weight=weights)
        # print(loss)
        return wce


class WeightedDiceLoss(Module):
    """Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations.
    Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M. Jorge Cardoso"""

    def __init__(self):
        super().__init__()

    def forward(self, output, onehot_label):
        class_frequencies = torch.sum(onehot_label, dim=(0, 2, 3))
        weights = torch.div(1., torch.add(torch.pow(class_frequencies, 2.), 1)).to(output.device)
        numerator = torch.sum(torch.mul(torch.sum(torch.mul(output, onehot_label), dim=(0, 2, 3)), weights))
        denominator = torch.sum(torch.mul(torch.sum(onehot_label + output, dim=(0, 2, 3)), weights))
        loss = -torch.log(torch.mul(torch.div(numerator, denominator), 2.0))
        print(loss)

        return loss


def custom_loss(output, onehot_label):
    wdc = WeightedDiceLoss()
    wce = WeightedCrossEntropyLoss()
    return wdc(output, onehot_label) * .2 + wce(output, onehot_label) * .8
