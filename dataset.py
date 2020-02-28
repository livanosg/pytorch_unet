import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
import pydicom as pd
from cv2 import warpAffine, getRotationMatrix2D, flip, filter2D, GaussianBlur
from config import paths
from helper_fns import one_hot


def augm_fn(image, label):
    """ Define the available methods for data augmentation and the available sets of these methods.
    Inputs:
        dcm_image -> pixel array of input data
        grd_image -> pixel array of label for given dcm_image
        augm_set -> one of ['geom', 'dist', 'all']
    Outputs:
        A tuple of (dcm_image, grd_image) after applying a data augmentation method,
        randomly chosen from the available sets."""

    # noinspection PyShadowingNames
    def rotate(input_image, label):
        angle = np.random.randint(-45, 45)
        rows, cols = input_image.shape
        m = getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)
        return warpAffine(input_image, m, (cols, rows)), warpAffine(label, m, (cols, rows))

    # noinspection PyShadowingNames
    def flips(input_image, label):
        flip_flag = np.random.randint(-1, 2)
        return flip(input_image, flip_flag), flip(label, flip_flag)

    # noinspection PyShadowingNames
    def s_n_p(input_image, label):
        p, b = 0.2, 0.05
        max_val = np.max(input_image)
        num_salt = np.ceil(b * input_image.size * p)
        coords = tuple([np.random.randint(0, dim - 1, int(num_salt)) for dim in input_image.shape])
        input_image[coords] = max_val
        num_pepper = np.ceil(b * input_image.size * (1. - p))
        coords = tuple([np.random.randint(0, dim - 1, int(num_pepper)) for dim in input_image.shape])
        input_image[coords] = 0
        return input_image, label

    # noinspection PyShadowingNames
    def sharp(input_image, label):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return filter2D(input_image, -1, kernel), label

    # noinspection PyShadowingNames
    def gaussian_blur(input_image, label):
        return GaussianBlur(input_image, (5, 5), 4), label

    # noinspection PyShadowingNames
    def contrast(input_image, label):
        contrast_factor = np.random.rand() * 2.
        image_mean = np.mean(input_image)
        image_contr = (input_image - image_mean) * contrast_factor + image_mean
        return image_contr, label

    # noinspection PyShadowingNames
    def random_translation(input_image, label):
        x = np.random.randint(-128, 129)
        y = np.random.randint(-128, 129)
        m = np.float32([[1, 0, x], [0, 1, y]])
        rows, cols, = input_image.shape
        return warpAffine(input_image, m, (cols, rows)), warpAffine(label, m, (cols, rows))

    # Random choice of augmentation method
    assert image.shape == label.shape
    all_processes = (rotate, flips, random_translation, s_n_p, sharp, gaussian_blur, contrast)
    augm = np.random.choice(all_processes[:3])
    dcm_image, grd_image = augm(image, label)
    augm = np.random.choice(all_processes[3:])
    return augm(dcm_image, grd_image)


def datasets(mode, branch_to_train=1):
    """Create a data generator for the given arguments
    Inputs:
    mode -> Modes same as tf.estimator.ModeKeys
    modality -> one of ['CT', 'MR', 'ALL']
    shuffle -> shuffle data before yielding
    augm_set -> one of ['geom', 'dist', 'all', 'none']
    augm_prob -> Values [0., 1.]. The augmentation probability if augm_set is not 'none'.
    only_paths -> yield only the paths of the data
    Outputs:
    A tuple of (dcm_image, grd_image) which is defined by the given arguments for a given dataset."""
    dicom_paths = glob(paths[mode] + '/CT/**/DICOM_anon/*.dcm', recursive=True)  # Get DICOM-data paths for given mode
    dicom_paths = sorted(dicom_paths)
    if mode in ('train', 'eval'):
        if branch_to_train == 1:
            ground_paths = glob(paths[mode] + '/CT/**/Ground/*.png', recursive=True)
        elif branch_to_train == 2:
            ground_paths = glob(paths[mode] + '/CT/**/Ground_2/*.png', recursive=True)
        else:
            return ValueError('Wrong branch value: %d' % branch_to_train)
        ground_paths = sorted(ground_paths)
        assert len(dicom_paths) == len(ground_paths)  # Check lists length
        zipped = list(zip(dicom_paths, ground_paths))
        # np.random.shuffle(zipped)
        return zipped
    if mode == 'infer':  # Yield only DICOM data for inference mode
        return dicom_paths


# noinspection PyUnresolvedReferences
class ChaosDataset(Dataset):
    """Chaos Dataset. Returns  input image and onehot labels """

    def __init__(self, mode, branch_to_train, num_classes, transform=None):
        """
        Args:
            mode (string): One of ['train', 'eval', 'pred'].
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode

        self.transform = transform
        self.num_classes = num_classes
        self.branch_to_train = branch_to_train
        self.samples = datasets(mode, branch_to_train=self.branch_to_train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        if self.mode in ('train', 'eval'):
            image = pd.dcmread(self.samples[index][0]).pixel_array.astype(np.float)
            label = cv2.imread(self.samples[index][1], 0)
            if self.mode == 'train' and self.transform:
                if np.random.random() < 0.5:  # Data augmentation
                    image, label = self.transform(image, label)  # data augmentation
            # Turn to pytorch Tensors
            image = torch.from_numpy(image).type(torch.float32)
            label = torch.from_numpy(label).type(torch.int16)

            # Normalize inputs
            image = (image - torch.mean(image)) / torch.std(image)
            image = torch.unsqueeze(image, dim=0)

            # For binary classes
            if self.branch_to_train == 1:
                if torch.max(label) == 0:
                    pass
                else:
                    label[label != torch.max(label)] = 0
                    label[label == torch.max(label)] = 1
                # onehot_label = one_hot(indices=label, num_classes=self.num_classes)
            else:
                if torch.max(label) == 0:
                    pass
                else:
                    label[label == torch.max(label)] = 2  # todo sanity check
                    label[label != torch.max(label) and label != torch.min(label)] = 1
            onehot_label = one_hot(indices=label, num_classes=self.num_classes)
            return {'input': image, 'label': onehot_label}

        if self.mode == 'infer':
            image = pd.dcmread(self.samples[index]).pixel_array.astype(np.float)
            image = (image - np.mean(image)) / np.std(image)  # normalize
            image = torch.from_numpy(image).type(torch.float32)
            image = torch.unsqueeze(image, dim=0)
            return {'input': image}


def dataloader(mode, branch_to_train, num_classes, batch_size):
    if mode == 'eval':
        augmentations = None
        shuffle = False
    else:
        augmentations = augm_fn
        shuffle = False
    return DataLoader(ChaosDataset(mode=mode, branch_to_train=branch_to_train, num_classes=num_classes, transform=augmentations), batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    pass
    # data = dataloader('train', 1, 2, 1)
    # for i in data:
    #     dcm = i['input']
    #     dcm = np.squeeze(dcm)
    #     label = i['label']
    #     label = np.argmax(label, 1)
    #     label = np.squeeze(label)
    #     cv2.imshow('', np.vstack((dcm, label)))
    #     cv2.waitKey()
