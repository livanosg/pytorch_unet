import os

import cv2
import numpy as np
import torch

from dataset import dataloader
from helper_fns import load_ckp
from models import Ynet

multi_gpus = False
branch_to_train = 1
dropout = 0.5
classes = 2
model = Ynet(branch_to_train=branch_to_train, dropout=dropout, output_classes=classes,
             split_gpus=multi_gpus)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

path = '/home/medphys/projects/pytorch/saves/ynet/best_model/best'
train_loader = dataloader(mode='train', branch_to_train=1, num_classes=2, batch_size=1)
model, optimizer, loaded_epoch, val_loss = load_ckp(path, model=model, optimizer=optimizer)
model.eval()
with torch.no_grad():
    for train_data in train_loader:  # step, batch
        inputs = train_data['input'].to('cpu')
        labels = train_data['label'].to('cpu')
        ground_path = train_data['ground_path'][0]
        print(ground_path)
        outputs = model(inputs)
        output = torch.squeeze(torch.argmax(outputs[0], dim=1)).cpu().numpy().astype(np.float32)
        labels = torch.squeeze(torch.argmax(labels, dim=1)).cpu().numpy().astype(np.float32)

        label_2 = labels + output * 2
        label_2[label_2 == 3] = 0
        os.makedirs(os.path.dirname(ground_path.replace('Ground', 'Ground_2')), exist_ok=True)
        cv2.imwrite(ground_path.replace('Ground', 'Ground_2'), label_2)
