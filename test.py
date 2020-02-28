import cv2
import torch
import numpy as np
import pydicom
from models import Ynet

load_path = 'C:\\Users\\livan\\PycharmProjects\\pytorch\\saves\\ynet\\checkpoint\\chkpt'
dcm_path = 'C:\\Users\\livan\\PycharmProjects\\pytorch\\Datasets\\Train_Sets\\CT\\18\\DICOM_anon\\i0033,0000b.dcm'


# Load model dict
def load_model(model, load_path, device):
    state_dict = torch.load(load_path, map_location=device)
    # Build model with parameters
    model.load_state_dict(state_dict)
    return model




