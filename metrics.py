import torch
from torch.nn import Module
from losses import eps


class DiceScore(Module):
    """Weighted Dice Loss computed from class-map label.
    dice_type: 'micro' 'macro' 'weighted'
    """

    def __init__(self, dice_type):
        super().__init__()
        self.dice_type = dice_type

    def forward(self, output, onehot_label):
        output_nograd = output.detach()
        numerator = torch.sum(torch.mul(input=onehot_label, other=output_nograd), dim=[0, 2, 3])
        denominator = torch.sum(torch.add(input=output_nograd, other=onehot_label), dim=[0, 2, 3])

        # F1 micro
        if self.dice_type == 'micro':
            f1 = torch.div(torch.mul(numerator, 2.0), torch.add(denominator, eps))

        # F1 macro
        elif self.dice_type == 'macro':
            f1 = torch.div(2.0 * torch.sum(numerator), torch.sum(torch.add(denominator, eps)))

        # F1 weighted
        elif self.dice_type == 'weighted':
            class_frequencies = torch.sum(onehot_label, dim=[0, 2, 3])  # Frequencies
            weights = torch.div(class_frequencies, torch.sum(class_frequencies))  # Correct F1 weighted
            f1 = torch.sum(torch.mul(torch.div(torch.mul(numerator, 2.0), torch.add(denominator, eps)), weights))
        else:
            return ValueError('Unknown F1 Score type: {}'.format(self.dice_type))
        return f1
