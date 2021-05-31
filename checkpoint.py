import os
import torch
import shutil

def save_checkpoint(state, is_best, epoch, save_path='./'):
    print("=> saving checkpoint '{}'".format(epoch))
    torch.save(state, os.path.join(save_path, 'checkpoint.pth.tar'))
    if(epoch % 10 == 0):
        torch.save(state, os.path.join(save_path, 'checkpoint_%03d.pth.tar' % epoch))
    if is_best:
        if epoch >= 90:
            shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'), 
                            os.path.join(save_path, 'model_best_in_100_epochs.pth.tar'))
        else:
            shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'), 
                            os.path.join(save_path, 'model_best_in_090_epochs.pth.tar'))


def load_checkpoint(args, model, optimizer=None, verbose=True):

    checkpoint = torch.load(args.resume)

    start_epoch = 0
    best_acc = 0

    if "epoch" in checkpoint:
        start_epoch = checkpoint['epoch']

    if "best_acc" in checkpoint:
        best_acc = checkpoint['best_acc']

    model.load_state_dict(checkpoint['state_dict'], False)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)

    if verbose:
        print("=> loading checkpoint '{}' (epoch {})"
                .format(args.resume, start_epoch))
    
    return model, optimizer, best_acc, start_epoch