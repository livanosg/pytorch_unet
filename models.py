import cv2
import torch
from torch import nn
from blocks import Encoder, Decoder


class Unet(nn.Module):
    def __init__(self, dropout, output_classes):
        super().__init__()
        self.encoder = Encoder(dropout=dropout)
        self.decoder = Decoder(dropout=dropout, branch=2, output_classes=output_classes)

    def forward(self, x):
        x, skip_1, skip_2, skip_3, skip_4 = self.encoder(x)
        output = self.decoder(x, skip_1, skip_2, skip_3, skip_4)
        return output


class Ynet(nn.Module):
    def __init__(self, branch_to_train, dropout, output_classes, split_gpus):
        super().__init__()
        self.encoder = Encoder(dropout=dropout)
        self.decoder = Decoder(dropout=dropout, branch=1, output_classes=output_classes)
        self.decoder_2 = Decoder(dropout=dropout, branch=2, output_classes=output_classes ** 2 - output_classes + 1)
        self.split_gpus = split_gpus
        if split_gpus:
            self.encoder.to("cuda:0")
            self.decoder.to("cuda:0")
            self.decoder_2.to("cuda:1")
        self.branch_to_train = branch_to_train
        if self.branch_to_train == 1:
            self.decoder_2.requires_grad = False
        if self.branch_to_train == 2:
            self.encoder.requires_grad = False
            self.decoder.requires_grad = False

    def forward(self, x):
        x, skip_11, skip_12, skip_13, skip_14 = self.encoder(x)
        output_1, skip_21, skip_22, skip_23, skip_24 = self.decoder(x, skip_11, skip_12, skip_13, skip_14)
        if self.split_gpus:
            x = x.to("cuda:1")  # P2P GPU transfer
            skip_21 = skip_21.to("cuda:1")
            skip_22 = skip_22.to("cuda:1")
            skip_23 = skip_23.to("cuda:1")
            skip_24 = skip_24.to("cuda:1")
        output_2 = self.decoder_2(x, skip_24, skip_23, skip_22, skip_21)
        return output_1, output_2

    def predict(self, inputs):
        self.eval()
        output_1, output_2 = self.forward(inputs)
        # Final Result
        # Get classes from 2nd output
        prediction = torch.argmax(output_1, dim=1)
        prediction = torch.squeeze(prediction)
        prediction_2 = torch.argmax(output_2, dim=1)
        prediction_2 = torch.squeeze(prediction_2)
        prediction[prediction_2 == 1] = 0  # Remove FP
        prediction[prediction_2 == 2] = 1  # Add FN
        return prediction

    def make_labels_2(self, inputs, label, label_path):
        """ Get output of branch_1 and create labels or the second branch.
        # Class mapping of second branch labels:
                                                0 = TP + TN
                                                1 = FP
                                                2 = FN
        inputs: input image in (batch, classes, Height, Width) shape. contains the prob of each class (Softmax output).
        label: labels of input in (batch, Height, Width) shape. Contains the correct class of each pixel.
        label_path: The path of current label"""
        self.eval()
        output_1, output_2 = self.forward(inputs)
        output_1 = torch.argmax(output_1, dim=1)
        label_2 = output_1 + label * 2
        label_2[label_2 == 3] = 0

        # Convert to png and save
        label_2 = label_2 * 255 // 3
        new_label_path = label_path.replace('Ground', 'Ground_2')
        cv2.imwrite(new_label_path, label_2)
# if __name__ == '__main__':
# model = Ynet(1, 0.5, 2)
#
# print(model)
