import argparse
from train import train_model

parser = argparse.ArgumentParser(description='Train a model according to given hyperparameters.')
parser.add_argument('-m', '--model', type=str, default='unet', choices=('unet', 'ynet'), help='Select model architecture.')
parser.add_argument('-branch', '--branch_to_train', type=int, default=1, choices=[1, 2], help='Set training branch in ynet.')
parser.add_argument('-dr', '--dropout', type=float, default=0.5, help='Learning Rate.')

parser.add_argument('-cls', '--classes', type=int, default=2, help='Choose between 2classes or 3 classes training.')
parser.add_argument('-b', '--batch', type=int, default=4, help='Define batch size per device')

parser.add_argument('-e', '--epochs', type=int, default=200, help='Total training epochs.')
parser.add_argument('-es', '--early_stopping_epochs', type=int, default=20, help='Early stopping epochs.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('-decay', '--decay', type=float, default=0.1, help='Decay of learning rate per decay_epochs. new_learning_rate = learning_rate * decay')
parser.add_argument('-de', '--decay_epochs', type=int, default=100, help='epochs to reach the learning_rate * decay value.')
parser.add_argument('-load', '--load_model', type=str, default='', help='The name of the folder a model is saved. If declared, The model will be loaded.')
parser.add_argument('-logs', '--logs_per_epoch', type=int, default=2, help='How many times will Log training metrics in every epoch.')


args = parser.parse_args()

if __name__ == '__main__':
    train_model(args)
