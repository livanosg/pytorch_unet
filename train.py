import os
import torch
from torch.utils.tensorboard import SummaryWriter
from config import paths
from helper_fns import metrics_writer, EarlyStopping
from metrics import DiceScore
from models import Ynet, Unet
from losses import custom_loss
from dataset import dataloader
from torch.optim.lr_scheduler import ExponentialLR


# noinspection DuplicatedCode
def train_model(args):
    multi_gpus = False
    total_batch = args.batch
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
    f1_micro = DiceScore(dice_type='micro')
    f1_macro = DiceScore(dice_type='macro')
    f1_weighted = DiceScore(dice_type='weighted')
    if args.model == 'unet':
        model = Unet(dropout=args.dropout, output_classes=args.classes)
        if multi_gpus:
            print('Model parallel distribution in {} GPUs'.format(torch.cuda.device_count()))
            total_batch = torch.cuda.device_count() * args.batch
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()),
                                          output_device=device)
        model.to(device)

    elif args.model == 'ynet':
        model = Ynet(branch_to_train=args.branch_to_train, dropout=args.dropout, output_classes=args.classes,
                     split_gpus=multi_gpus)
        if args.branch_to_train == 2:
            args.classes = args.classes ** 2 - args.classes + 1
        if multi_gpus:
            print('Model will be splitted in 2 GPUs')
        else:
            model.to(device)

    else:
        return NotImplementedError('Model {} is not implemented'.format(args.model))

    early_stopping = EarlyStopping(patience=args.early_stopping_epochs)

    # Declare data
    train_loader = dataloader(mode='train', branch_to_train=args.branch_to_train, num_classes=args.classes, batch_size=total_batch)
    val_loader = dataloader(mode='eval', branch_to_train=args.branch_to_train, num_classes=args.classes, batch_size=total_batch)

    # Declare optimizer and learning rate decay
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    gamma = args.decay ** (1 / args.decay_epochs)
    scheduler = ExponentialLR(optimizer, gamma=gamma, last_epoch=-1)

    # Steps to keep logs
    logs_step = len(train_loader) // args.logs_per_epoch

    # define model save path
    trial = 0
    model_path = paths['save_model'] + '/' + args.model
    while os.path.exists(model_path + '_trial_{}'.format(trial)):
        trial += 1
    model_root = model_path + '_trial_{}'.format(trial)

    # print characteristics
    print('Model will be saved in {}'.format(model_root))
    print('Training starts...')
    print('Train for {} epochs with {} early stopping epochs.'.format(args.epochs, args.early_stopping_epochs))
    print('learning rate is {} with exponential decay to {} of learning rate fraction every {} epochs.'.format(
        args.learning_rate, args.decay, args.decay_epochs))
    print('{} steps per epoch with {} samples per batch.'.format(len(train_loader), total_batch))
    print('logs every {} steps.'.format(logs_step))
    print('{} GPUs utilized.'.format(torch.cuda.device_count()))
    writer = SummaryWriter(model_root + '/' + 'events/')
    global_step = 0
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_f1_micro = 0.0
        running_f1_macro = 0.0
        running_f1_weighted = 0.0
        model.train()
        for train_step, train_data in enumerate(train_loader):  # step, batch
            global_step = epoch * len(train_loader) + train_step + 1
            optimizer.zero_grad()  # zero the parameter gradients
            train_inputs = train_data['input'].to(device)
            train_labels = train_data['label'].to(device)
            train_outputs = model(train_inputs)

            if args.model == 'ynet':  # select proper output to propagate loss for correct branch
                train_outputs = train_outputs[args.branch_to_train - 1]
                train_outputs = train_outputs.to(device)
            # forward + backward + optimize
            train_loss = custom_loss(train_outputs, train_labels)
            train_loss.backward()
            optimizer.step()  # Update variables

            # Metrics
            train_f1_micro = f1_micro(train_outputs, train_labels)
            train_f1_macro = f1_macro(train_outputs, train_labels)
            train_f1_weighted = f1_weighted(train_outputs, train_labels)

            running_loss += train_loss.item()
            running_f1_micro += train_f1_micro
            running_f1_macro += train_f1_macro.item()
            running_f1_weighted += train_f1_weighted.item()

            metrics = {'loss': running_loss / (train_step + 1),
                       'f1_micro': running_f1_micro / (train_step + 1),
                       'f1_macro': running_f1_macro / (train_step + 1),
                       'f1_weighted': running_f1_weighted / (train_step + 1),
                       'lr': scheduler.get_lr()[0]}

            if global_step % 50 == 0 or global_step == 1 or (global_step % (len(train_loader) // 2) == 0):
                print('Train -->  Epoch: {0:3d}, Step: {1:5d} Loss: {2:.4f}'.format(epoch + 1, global_step,
                                                                                    metrics['loss']))
                print('           Background       Liver')
                print('F1-micro: {} F1-macro: {}, (Dice) F1-weighted: {}%'.format(metrics['f1_micro'].cpu().numpy(),
                                                                                  metrics['f1_macro'],
                                                                                  metrics['f1_weighted'] * 100))
            if (global_step % (len(train_loader) // 2) == 0) or (global_step == 1):
                metrics_writer(writer=writer, mode='train', input=train_inputs, onehot_label=train_labels,
                               output=train_outputs, metrics=metrics, model=model, global_step=global_step)
        running_loss = 0.0
        running_f1_micro = 0.0
        running_f1_macro = 0.0
        running_f1_weighted = 0.0
        model.eval()
        with torch.no_grad():
            for val_step, val_data in enumerate(val_loader):
                val_inputs = val_data['input'].to(device)
                val_labels = val_data['label'].to(device)
                val_outputs = model(val_inputs)

                if args.model == 'ynet':
                    val_outputs = val_outputs[args.branch_to_train - 1]
                    val_outputs = val_outputs.to(device)

                val_loss = custom_loss(val_outputs, val_labels).item()
                running_loss += val_loss

                val_f1_micro = f1_micro(val_outputs, val_labels)
                val_f1_macro = f1_macro(val_outputs, val_labels)
                val_f1_weighted = f1_weighted(val_outputs, val_labels)
                running_f1_micro += val_f1_micro
                running_f1_macro += val_f1_macro
                running_f1_weighted += val_f1_weighted

                val_metrics = {'loss': running_loss / (val_step + 1),
                               'f1_micro': running_f1_micro / (val_step + 1),
                               'f1_macro': running_f1_macro / (val_step + 1),
                               'f1_weighted': running_f1_weighted / (val_step + 1)}

            print('Eval -->  Epoch: {0:3d}, Step: {1:5d} Loss: {2:.4f}'.format(epoch + 1, global_step,
                                                                               val_metrics['loss']))
            print('           Background       Liver')
            print('F1-micro: {} F1-macro: {}, (Dice) F1-weighted: {}%'.format(val_metrics['f1_micro'],
                                                                              val_metrics['f1_macro'],
                                                                              val_metrics['f1_weighted']))
            checkpoint = {'epoch': epoch + 1,
                          'valid_loss_min': val_metrics['loss'],
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            metrics_writer(writer=writer, mode='eval', input=val_inputs, onehot_label=val_labels,
                           output=val_outputs, metrics=val_metrics, model=model, global_step=global_step)

        early_stopping(val_metric=val_metrics['loss'], minimize=True, checkpoint=checkpoint, model_root=model_root, epoch=epoch)
        # noinspection PyArgumentList
        scheduler.step()
    writer.close()
    print('Training finished without facing early stopping criteria')
