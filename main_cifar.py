from __future__ import absolute_import

# system lib
import os
import time
import sys
import argparse
# numerical libs
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
# models
from thop import profile
from util import AverageMeter, ProgressMeter, accuracy, parse_gpus
from checkpoint import save_checkpoint, load_checkpoint
from networks.cifar import create_net


def adjust_learning_rate(optimizer, epoch, warmup=False):
    """Adjust the learning rate"""
    if epoch <= 81:
        lr = 0.01 if warmup and epoch == 0 else args.base_lr
    elif epoch <= 122:
        lr = args.base_lr * 0.1
    else:
        lr = args.base_lr * 0.01

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(net, optimizer, epoch, data_loader, args):

    learning_rate = optimizer.param_groups[0]["lr"]

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':4.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch (Train LR {:6.4f}): [{}] ".format(learning_rate, epoch))

    net.train()

    tic = time.time()
    for batch_idx, (data, target) in enumerate(data_loader):
     
        data, target = data.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)

        data_time.update(time.time() - tic)

        optimizer.zero_grad()
        output = net(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        acc = accuracy(output, target)
        losses.update(loss.item(), data.size(0))
        top1.update(acc[0].item(), data.size(0))

        batch_time.update(time.time() - tic)
        tic = time.time()
        
        if (batch_idx+1) % args.disp_iter == 0 or (batch_idx+1) == len(data_loader):
            epoch_msg = progress.get_message(batch_idx+1)
            print(epoch_msg)

            args.log_file.write(epoch_msg + "\n")

def validate(net, epoch, data_loader, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':4.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch (Valid LR {:6.4f}): [{}] ".format(0, epoch))

    net.eval()

    with torch.no_grad():
        tic = time.time()
        for batch_idx, (data, target) in enumerate(data_loader):
        
            data, target = data.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)

            data_time.update(time.time() - tic)

            output = net(data)
            loss = F.cross_entropy(output, target)

            acc = accuracy(output, target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc[0].item(), data.size(0))

            batch_time.update(time.time() - tic)
            tic = time.time()
            
            if (batch_idx+1) % args.disp_iter == 0 or (batch_idx+1) == len(data_loader):
                epoch_msg = progress.get_message(batch_idx+1)
                print(epoch_msg)

                args.log_file.write(epoch_msg + "\n")
    
        print('-------- Mean Accuracy {top1.avg:.3f} --------'.format(top1=top1))

    return top1.avg

def main(args):

    if len(args.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        cudnn.benchmark = True
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        args.device = torch.device("cuda:{}".format(args.gpu_ids[0]))
    else:
        kwargs = {}
        args.device = torch.device("cpu")

    normlizer = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    print("Building dataset: " + args.dataset)

    if args.dataset == "cifar10":
        args.num_class = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataset_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normlizer])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataset_dir, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            normlizer])),
            batch_size=100, shuffle=False, **kwargs)

    elif args.dataset == "cifar100":
        args.num_class = 100
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.dataset_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normlizer])), 
                        batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.dataset_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normlizer])),
                batch_size=100, shuffle=False, **kwargs)

    net = create_net(args)

    print(net)

    optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=args.beta1, weight_decay=args.weight_decay)

    if args.resume:
        net, optimizer, best_acc, start_epoch = load_checkpoint(args, net, optimizer)
    else:
        start_epoch = 0
        best_acc = 0

    x = torch.randn(1, 3, 32, 32)
    flops, params = profile(net, inputs=(x,))

    print("Number of params: %.6fM" % (params / 1e6))
    print("Number of FLOPs: %.6fG" % (flops / 1e9))

    args.log_file.write("Network - " + args.arch + "\n")
    args.log_file.write("Attention Module - " + args.attention_type + "\n")
    args.log_file.write("Params - %.6fM" % (params / 1e6) + "\n")
    args.log_file.write("FLOPs - %.6fG" % (flops / 1e9) + "\n")
    args.log_file.write("--------------------------------------------------" + "\n")

    if len(args.gpu_ids) > 0:
        net.to(args.gpu_ids[0])
        net = torch.nn.DataParallel(net, args.gpu_ids)  # multi-GPUs

    for epoch in range(start_epoch, args.num_epoch):
        # if args.wrn:
            # adjust_learning_rate_wrn(optimizer, epoch, args.warmup)
        # else:
        adjust_learning_rate(optimizer, epoch, args.warmup)

        train(net, optimizer, epoch, train_loader, args)
        epoch_acc = validate(net, epoch, test_loader, args)

        is_best = epoch_acc > best_acc
        best_acc = max(epoch_acc, best_acc)

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": net.module.cpu().state_dict(),
            "best_acc": best_acc,
            "optimizer" : optimizer.state_dict(),
            }, is_best, epoch, save_path=args.ckpt)

        net.to(args.device)

        args.log_file.write("--------------------------------------------------" + "\n")

    args.log_file.write("best accuracy %4.2f" % best_acc)

    print("Job Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CIFAR baseline")

    # Model settings
    parser.add_argument("--arch", type=str, default="resnet18",
                        help="network architecture (default: resnet18)")
    parser.add_argument("--num_base_filters", type=int, default=16, 
                        help="network base filer numbers (default: 16)")
    parser.add_argument("--expansion", type=float, default=1,
                        help="expansion factor for the mid-layer in resnet-like")
    parser.add_argument("--block_type", type=str, default="basic", 
                        help="building block for network, e.g., basic or bottlenect")
    parser.add_argument("--attention_type", type=str, default="none",
                        help="attention type in building block (possible choices none | se | cbam | simam )")
    parser.add_argument("--attention_param", type=float, default=4,
                        help="attention parameter (reduction in CBAM and SE, e_lambda in simam)")
    
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="training dataset (default: cifar10)")
    parser.add_argument("--dataset_dir", type=str, default="data",
                        help="data set path (default: data)")
    parser.add_argument("--workers", default=16, type=int, 
                        help="number of data loading works")

    # Optimizion settings
    parser.add_argument("--gpu_ids", default="0",
                        help="gpus to use, e.g. 0-3 or 0,1,2,3")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size for training and validation (default: 128)")
    parser.add_argument("--num_epoch", type=int, default=164,
                        help="number of epochs to train (default: 164)")
    parser.add_argument("--resume", default="", type=str,
                        help="path to checkpoint for continous training (default: none)")
    parser.add_argument("--optim", default="SGD", 
                        help="optimizer")
    parser.add_argument("--base_lr", type=float, default=0.1, 
                        help="learning rate (default: 0.1)")
    parser.add_argument("--beta1", default=0.9, type=float,
                        help="momentum for sgd, beta1 for adam")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="SGD weight decay (default: 5e-4)")
    parser.add_argument("--warmup", action="store_true",
                        help="warmup for deeper network")
    parser.add_argument("--wrn", action="store_true",
                        help="wider resnet for training")
    
    # Misc
    parser.add_argument("--seed", type=int, default=1, 
                        help="random seed (default: 1)")
    parser.add_argument("--disp_iter", type=int, default=100,
                        help="frequence to display training status (default: 100)")
    parser.add_argument("--ckpt", default="./ckpts/", 
                        help="folder to output checkpoints")

    args = parser.parse_args()
    args.gpu_ids = parse_gpus(args.gpu_ids)

    args.ckpt += args.dataset
    args.ckpt += "-" + args.arch
    args.ckpt += "-" + args.block_type
    if args.attention_type.lower() != "none":
        args.ckpt += "-" + args.attention_type
    if args.attention_type.lower() != "none":
        args.ckpt += "-param" + str(args.attention_param)
    args.ckpt += "-nfilters" + str(args.num_base_filters)
    args.ckpt += "-expansion" + str(args.expansion)
    args.ckpt += "-baselr" + str(args.base_lr)
    args.ckpt += "-rseed" + str(args.seed)
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))
 
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    # write to file
    args.log_file = open(os.path.join(args.ckpt, "log_file.txt"), mode="w")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
 
    main(args)

    args.log_file.close()
