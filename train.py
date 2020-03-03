import os
import torch
from torch.utils.tensorboard import SummaryWriter
from config import paths
from helper_fns import metrics_writer, EarlyStopping, load_ckp
from metrics import DiceScore
from models import Ynet, Unet
from losses import custom_loss
from dataset import dataloader
from torch.optim.lr_scheduler import ExponentialLR


# noinspection DuplicatedCode

def distr_train(args):
    multi_gpus = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print('Preparing training of {} model.'.format(args.model))
    if torch.cuda.is_available():
        print(torch.cuda.device_count(), " available GPUs!")
        device = torch.device("cuda:0")
        if torch.cuda.device_count() > 1:
            multi_gpus = True
    else:
        print('Running on CPU...')
        device = torch.device("cpu")
    return device, multi_gpus


def get_model(args, device, multi_gpus):
    if args.model == 'unet':
        model = Unet(dropout=args.dropout, output_classes=args.classes)
        batch = args.batch
        if multi_gpus:
            print('Model parallel distribution in {} GPUs'.format(torch.cuda.device_count()))
            batch = torch.cuda.device_count() * args.batch
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()),
                                          output_device=device)
        model.to(device)
    elif args.model == 'ynet':
        model = Ynet(branch_to_train=args.branch_to_train, dropout=args.dropout, output_classes=args.classes,
                     split_gpus=multi_gpus)
        batch = args.batch
        if multi_gpus:
            print('Model will be splitted in 2 GPUs')
        else:
            model.to(device)
    else:
        return NotImplementedError('Model {} is not implemented'.format(args.model))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    return model, optimizer,  batch


def get_dataloaders(args, batch):
    if args.model == 'unet':
        branch_to_train = 1
    else:
        branch_to_train = args.branch_to_train
    train_loader = dataloader(mode='train', branch_to_train=branch_to_train, batch_size=batch, classes=args.classes)
    val_loader = dataloader(mode='eval', branch_to_train=branch_to_train, batch_size=batch, classes=args.classes)
    return train_loader, val_loader


def load_set_model(args, model, optimizer):
    if args.load_model:
        load_path = paths['save_model'] + '/' + args.load_model
        model_path = os.path.split(load_path)[0]
        model, optimizer, loaded_epoch, val_loss = load_ckp(load_path, model=model, optimizer=optimizer)
        return model, model_path, optimizer, loaded_epoch, val_loss
    else:
        trial = 0
        while os.path.exists(paths['save_model'] + '/' + args.model + '{}'.format(trial)):
            trial += 1
        model_path = paths['save_model'] + '/' + args.model + '{}'.format(trial)
        return model, model_path, optimizer


def exp_dec_schedule(args, optimizer):
    gamma = args.decay ** (1 / args.decay_epochs)
    scheduler = ExponentialLR(optimizer, gamma=gamma, last_epoch=-1)
    return scheduler


def f1_metrics(outputs, labels):
    f1_micro = DiceScore(dice_type='micro')
    f1_macro = DiceScore(dice_type='macro')
    f1_weighted = DiceScore(dice_type='weighted')

    f1_micro = f1_micro(outputs, labels)
    f1_macro = f1_macro(outputs, labels)
    f1_weighted = f1_weighted(outputs, labels)
    return f1_micro, f1_macro, f1_weighted


def logs(global_step, data_size, epoch, metrics, input, label, output, model, mode, writer):
    if global_step % 50 == 0 or global_step == 1 or (global_step % (data_size // 2) == 0):
        print('Train -->  Epoch: {0:3d}, Step: {1:5d} Loss: {2:.3f}'.format(epoch + 1, global_step,
                                                                            metrics['loss']))
        print('Background'.rjust(20, ' '), 'Liver'.rjust(6, ' '))
        print('F1-micro: {0[0]:5.3f} {0[1]:11.3f}\nF1-macro: {1:10.3f}\nF1-weighted: {2:7.3f}'.format(
            metrics['f1_micro'],
            metrics['f1_macro'],
            metrics['f1_weighted']))
    if (global_step % (data_size // 2) == 0) or (global_step == 1):
        metrics_writer(writer=writer, mode=mode, input=input, onehot_label=label,
                       output=output, metrics=metrics, model=model, global_step=global_step)


def epoch_pass(args, mode, model, optimizer, scheduler, device, data_loader, current_epoch, writer, early_stop):
    # Metrics
    running_loss = 0.0
    running_f1_micro = 0.0
    running_f1_macro = 0.0
    running_f1_weighted = 0.0
    data_size = len(data_loader)
    if mode == 'train':
        model.train()
    if mode == 'eval':
        model.eval()
    for step, data in enumerate(data_loader):  # step, batch
        global_step = current_epoch * len(data_loader) + step + 1
        if mode == 'train':
            optimizer.zero_grad()  # zero the parameter gradients
        input = data['input'].to(device)
        label = data['label'].to(device)
        output = model(input)
        if args.model == 'ynet':  # select proper output to propagate loss for correct branch
            output = output[args.branch_to_train - 1].to(device)
        # forward + backward + optimize
        loss = custom_loss(output, label)
        if mode == 'train':
            loss.backward()
            optimizer.step()  # Update variables

        f1_micro, f1_macro, f1_weighted = f1_metrics(output, label)

        running_loss += loss
        running_f1_micro += f1_micro
        running_f1_macro += f1_macro
        running_f1_weighted += f1_weighted
        metrics = {'loss': running_loss / (step + 1),
                   'f1_micro': running_f1_micro / (step + 1),
                   'f1_macro': running_f1_macro / (step + 1),
                   'f1_weighted': running_f1_weighted / (step + 1),
                   'lr': scheduler.get_lr()[0]}
        logs(global_step=global_step, mode=mode, data_size=data_size, epoch=current_epoch, metrics=metrics, input=input, label=label, output=output, model=model, writer=writer)
        if mode == 'eval':
            checkpoint = {'epoch': current_epoch + 1,
                          'valid_loss_min': metrics['loss'],
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            early_stop(val_metric=metrics['loss'], minimize=True, checkpoint=checkpoint, epoch=current_epoch)


def train_model(args):
    device, multi_gpus = distr_train(args)
    model, optimizer, batch = get_model(args, device, multi_gpus)
    train_loader, val_loader = get_dataloaders(args, batch)
    if args.load_model:
        model, model_path, optimizer, loaded_epoch, val_loss = load_set_model(args, model, optimizer)
    else:
        model, model_path, optimizer = load_set_model(args, model, optimizer)
        loaded_epoch = 0
    scheduler = exp_dec_schedule(args, optimizer)
    early_stopping = EarlyStopping(model_path=model_path, patience=args.early_stopping_epochs)
    writer = SummaryWriter(model_path + '/' + 'events/')

    print('Model folder is {}'.format(model_path))
    print('Training starts...')
    print('Train for {} epochs with {} early stopping epochs.'.format(args.epochs, args.early_stopping_epochs))
    print('learning rate is {} with exponential decay to {} of learning rate fraction every {} epochs.'.format(args.learning_rate, args.decay, args.decay_epochs))
    print('{} steps per epoch with {} samples per batch.'.format(len(train_loader), batch))
    print('{} GPUs utilized.'.format(torch.cuda.device_count()))

    for current_epoch in range(loaded_epoch, args.epochs):  # loop over the dataset multiple times
        epoch_pass(args, 'train', model, optimizer, scheduler, device, train_loader, current_epoch, writer, early_stopping)
        epoch_pass(args, 'eval', model, optimizer, scheduler, device, val_loader, current_epoch, writer, early_stopping)
        scheduler.step()
    writer.close()
    print('Training finished without facing early stopping criteria')
