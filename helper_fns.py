import os
import shutil
import torch


# noinspection PyShadowingBuiltins
def metrics_writer(writer, mode, input, onehot_label, output, metrics, model, global_step):
    input = (input - input.min()) / (input.max() - input.min())
    output = torch.unsqueeze(torch.argmax(output, dim=1), dim=1)
    onehot_label = torch.unsqueeze(torch.argmax(onehot_label, dim=1), dim=1)

    writer.add_images('{}/Input'.format(mode), torch.unsqueeze(input[0, :, :, :], 0), global_step=global_step)
    writer.add_images('{}/Label'.format(mode), torch.unsqueeze(onehot_label[0, :, :, :], 0), global_step=global_step)
    writer.add_images('{}/Output'.format(mode), torch.unsqueeze(output[0, :, :, :], 0), global_step=global_step)
    writer.add_scalars('Loss', {str(mode): metrics['loss']}, global_step=global_step)
    writer.add_scalars('f1_micro-background', {mode: metrics['f1_micro'][0]}, global_step=global_step)
    writer.add_scalars('f1_micro-liver', {mode: metrics['f1_micro'][1]}, global_step=global_step)
    writer.add_scalars('f1_macro', {mode: metrics['f1_macro']}, global_step=global_step)
    writer.add_scalars('f1_weighted', {mode: metrics['f1_weighted']}, global_step=global_step)
    if mode == 'train':
        writer.add_scalar('Learning Rate', metrics['lr'], global_step=global_step)
    if global_step == 0:
        writer.add_graph(model, input)
    writer.flush()


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, checkpoint_path)
    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(checkpoint_path, best_model_path)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, model_path, patience=7, counter=0, delta=0, ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = counter
        self.delta = delta
        self.metric_best = torch.tensor(-float('inf'))
        self.model_path = model_path

    def __call__(self, val_metric, checkpoint, epoch, minimize):
        self.counter += 1
        checkpoint_path = self.model_path + '/checkpoint{}'.format(epoch)
        best_model_path = self.model_path + '/best'

        if minimize:
            val_metric = -val_metric

        if val_metric > self.metric_best + self.delta:
            self.counter = 0
            self.metric_best = val_metric
            print(f'Early Stopping counter: {self.counter} out of {self.patience}')
            print('Current model is the best. Saving to: {}'.format(best_model_path))
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        else:
            print(f'Early Stopping counter: {self.counter} out of {self.patience}')
            print('Saving model to: {}'.format(checkpoint_path))
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        if self.counter >= self.patience:
            print('Early Stopping criteria reached.')
            print('Performance did not improve for the last {} epochs.'.format(self.counter))
            print('Training Finished')
            exit()
