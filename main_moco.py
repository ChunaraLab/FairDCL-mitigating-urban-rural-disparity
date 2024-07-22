#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import gc

import moco.builder
import moco.loader
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import moco.loader
import moco.builder

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=16,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.002,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=100,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=8192, type=int,
                    help='queue size; number of negative keys (default: 32 * 2^n)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def compute_cat(q_l, joint_ql, bs):
    new_batch_joint = []
    new_batch_margin = []
    for i in range(len(qs)):
        new_batch_joint.append(torch.cat((q_l[i], joint_ql[i]), 1))

        # marginal_q = 
    # new_batch_joint2 = torch.cat((q_l2, joint_q2), 1)
    # new_batch_joint3 = torch.cat((q_l3, joint_q3), 1)
    # new_batch_joint4 = torch.cat((q_l4, joint_q4), 1)
    # new_batch_joint5 = torch.cat((q, joint_q), 1)
    
    marginal_q1 = torch.permute(F.one_hot(torch.randint(low=0, high=2, size=(32, 1, 1))).expand(-1, 128, 128, -1), (0,3,1,2)).cuda()
    marginal_q2 = torch.permute(F.one_hot(torch.randint(low=0, high=2, size=(32, 1, 1))).expand(-1, 64, 64, -1), (0,3,1,2)).cuda()
    marginal_q3 = torch.permute(F.one_hot(torch.randint(low=0, high=2, size=(32, 1, 1))).expand(-1, 32, 32, -1), (0,3,1,2)).cuda()
    marginal_q4 = torch.permute(F.one_hot(torch.randint(low=0, high=2, size=(32, 1, 1))).expand(-1, 16, 16, -1), (0,3,1,2)).cuda()
    marginal_q5 = F.one_hot(torch.randint(low=0, high=2, size=(1, 32)))[0].cuda()

    new_batch_marginal1 = torch.cat((q_l1, marginal_q1), 1)
    new_batch_marginal2 = torch.cat((q_l2, marginal_q2), 1)
    new_batch_marginal3 = torch.cat((q_l3, marginal_q3), 1)
    new_batch_marginal4 = torch.cat((q_l4, marginal_q4), 1)
    new_batch_marginal5 = torch.cat((q, marginal_q5), 1)

    new_batch1 = torch.cat((new_batch_joint1, new_batch_marginal1), 0)
    new_batch2 = torch.cat((new_batch_joint2, new_batch_marginal2), 0)
    new_batch3 = torch.cat((new_batch_joint3, new_batch_marginal3), 0)
    new_batch4 = torch.cat((new_batch_joint4, new_batch_marginal4), 0)
    new_batch5 = torch.cat((new_batch_joint5, new_batch_marginal5), 0)
    
    return new_batch1, new_batch2, new_batch3, new_batch4, new_batch5
    


def main_worker(gpu, ngpus_per_node, args):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # # initialize the process group
    # dist.init_process_group("gloo", rank=0, world_size=1)

    using_discriminators = True

    # create model
    print("=> creating model '{}'".format(args.arch))

    resnet50 = models.__dict__[args.arch](pretrained=True)
    resnet50.fc = nn.Linear(2048, args.moco_dim)
    resnet50.train()

    resnet50_2 = models.__dict__[args.arch](pretrained=True)
    resnet50_2.fc = nn.Linear(2048, args.moco_dim)
    resnet50_2.train()

    model = moco.builder.MoCo(
        resnet50, resnet50_2, 
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, using_discriminators)


    urban = list(np.load('/scratch/mz2466/LoveDA_mixedD_moco/urban_ids.npy'))
    rural = list(np.load('/scratch/mz2466/LoveDA_mixedD_moco/rural_ids.npy'))

    urban_ids = {}
    rural_ids = {}
    for name in urban:
        urban_ids[name] = [1, 0]
    for name in rural:
        rural_ids[name] = [0, 1]

    #female_ids = set(np.load('/scratch/mz2466/Face/female_id.npy'))
    #male_ids = set(np.load('/scratch/mz2466/Face/male_id.npy'))
    #Mine = moco.builder.Mine(input_size=2050, hidden_size1=2050, hidden_size2=200)
    #Mine3 = moco.builder.Mine3(input_size=1026, hidden_size1=1026, hidden_size2=100)
    Discriminator1 = moco.builder.Discriminator1(input_size=258, hidden_size1=512, hidden_size2=512)
    Discriminator2 = moco.builder.Discriminator1(input_size=514, hidden_size1=512, hidden_size2=512)
    Discriminator3 = moco.builder.Discriminator1(input_size=1026, hidden_size1=512, hidden_size2=512)
    Discriminator4 = moco.builder.Discriminator1(input_size=2050, hidden_size1=512, hidden_size2=512)
    Global1 = moco.builder.Global1(input_size=514, hidden_size1=512, hidden_size2=512)

    model.cuda()
    Discriminator1.cuda()
    Discriminator2.cuda()
    Discriminator3.cuda()
    Discriminator4.cuda()
    Global1.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_d1 = torch.optim.Adam(Discriminator1.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    optimizer_d2 = torch.optim.Adam(Discriminator2.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    optimizer_d3 = torch.optim.Adam(Discriminator3.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    optimizer_d4 = torch.optim.Adam(Discriminator4.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    optimizer_d5 = torch.optim.Adam(Global1.parameters(), args.lr,
                                weight_decay=args.weight_decay)



    # optionally resume from a checkpoint
    if args.resume:
        ee = '30'
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint1 = torch.load('checkpoint_d1_00'+ee+'.pth.tar')
        checkpoint2 = torch.load('checkpoint_d2_00'+ee+'.pth.tar')
        checkpoint3 = torch.load('checkpoint_d3_00'+ee+'.pth.tar')
        checkpoint4 = torch.load('checkpoint_d4_00'+ee+'.pth.tar')
        checkpoint5 = torch.load('checkpoint_d5_00'+ee+'.pth.tar')
        Discriminator1.load_state_dict(checkpoint1['state_dict'])
        Discriminator2.load_state_dict(checkpoint2['state_dict'])
        Discriminator3.load_state_dict(checkpoint3['state_dict'])
        Discriminator4.load_state_dict(checkpoint4['state_dict'])
        Global1.load_state_dict(checkpoint5['state_dict'])
        optimizer_d1.load_state_dict(checkpoint1['optimizer'])
        optimizer_d2.load_state_dict(checkpoint2['optimizer'])
        optimizer_d3.load_state_dict(checkpoint3['optimizer'])
        optimizer_d4.load_state_dict(checkpoint4['optimizer'])
        optimizer_d5.load_state_dict(checkpoint5['optimizer'])

        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, '')
    normalize = transforms.Normalize(mean=[76.4951, 82.3195, 78.7478],
                                     std=[36.1791, 28.9031, 32.6070])

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(512),
            #transforms.RandomApply([
            #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            #], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            # transforms.RandomResizedCrop(512),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    train_dataset = moco.loader.TwoCropsTransform(traindir, transforms.Compose(augmentation))
    train_dataset_d = moco.loader.TwoCropsTransform(traindir, transforms.Compose(augmentation))


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True)

    train_loader_d = torch.utils.data.DataLoader(
        train_dataset_d, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True)


    Mine_train_loss_epoch = []
    Mine_loss_epoch = []
    SL_loss_epoch = []

    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, train_loader_d, train_dataset_d, model, Discriminator1, Discriminator2, Discriminator3, Discriminator4, Global1, criterion, optimizer, optimizer_d1, optimizer_d2, optimizer_d3, optimizer_d4, optimizer_d5, epoch, urban_ids, rural_ids, using_discriminators, Mine_train_loss_epoch, Mine_loss_epoch, SL_loss_epoch, args)

        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))
            
            if (using_discriminators):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': Discriminator1.state_dict(),
                    'optimizer' : optimizer_d1.state_dict(),
                }, is_best=False, filename='checkpoint_d1_{:04d}.pth.tar'.format(epoch))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': Discriminator2.state_dict(),
                    'optimizer' : optimizer_d2.state_dict(),
                }, is_best=False, filename='checkpoint_d2_{:04d}.pth.tar'.format(epoch))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': Discriminator3.state_dict(),
                    'optimizer' : optimizer_d3.state_dict(),
                }, is_best=False, filename='checkpoint_d3_{:04d}.pth.tar'.format(epoch))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': Discriminator4.state_dict(),
                    'optimizer' : optimizer_d4.state_dict(),
                }, is_best=False, filename='checkpoint_d4_{:04d}.pth.tar'.format(epoch))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': Global1.state_dict(),
                    'optimizer' : optimizer_d5.state_dict(),
                }, is_best=False, filename='checkpoint_d5_{:04d}.pth.tar'.format(epoch))


def train(train_loader, train_loader_d, train_dataset_d, model, Discriminator1, Discriminator2, Discriminator3, Discriminator4, Global1, criterion, optimizer, optimizer_d1, optimizer_d2, optimizer_d3, optimizer_d4, optimizer_d5, epoch, urban_ids, rural_ids, using_discriminators, Mine_train_loss_epoch, Mine_loss_epoch, SL_loss_epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    Mine_train_loss = []
    Mine_loss = []
    SL_loss = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, joint_q1, joint_q2, joint_q3, joint_q4, joint_q) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

        model.train()

        if (using_discriminators):
            # Training discriminators
            Discriminator1.train()
            Discriminator2.train()
            Discriminator3.train()
            Discriminator4.train()
            Global1.train()
            running_loss, running_mst = 0, 0
            for j, (images_d, joint_q1_d, joint_q2_d, joint_q3_d, joint_q4_d, joint_q_d) in enumerate(train_loader_d):
        
                images_d[0] = images_d[0].cuda(args.gpu, non_blocking=True)
                images_d[1] = images_d[1].cuda(args.gpu, non_blocking=True)
                if (j > 20):
                    train_loader_d = torch.utils.data.DataLoader(train_dataset_d, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True)
                    break
                q_l1, q_l2, q_l3, q_l4, backbone_feature = model(im_q=images_d[0], im_k=images_d[1], intermediate=True)
            
                new_batch1, new_batch2, new_batch3, new_batch4, new_batch5 = compute_cat(q_l1, q_l2, q_l3, q_l4, backbone_feature, joint_q1_d.cuda(), joint_q2_d.cuda(), joint_q3_d.cuda(), joint_q4_d.cuda(), joint_q_d.cuda())
                
                
                loss_m1, mst_m1 = Discriminator1(new_batch1)
                loss_m2, mst_m2 = Discriminator2(new_batch2)
                loss_m3, mst_m3 = Discriminator3(new_batch3)
                loss_m4, mst_m4 = Discriminator4(new_batch4)
                
                loss_m5, mst_m5 = Global1(new_batch5)
                #print(str(mst_m + mst_m3 + mst_m2))

                running_loss = running_loss +loss_m1.item() +loss_m2.item() +loss_m3.item() +loss_m4.item() +loss_m5.item()
                running_mst = running_mst +mst_m1.item() +mst_m2.item() +mst_m3.item() +mst_m4.item() +mst_m5.item()
                #running_loss = loss_m5.item()
                #running_mst = mst_m5.item()
                
                optimizer_d1.zero_grad()
                loss_m1.backward()
                optimizer_d1.step()

                optimizer_d2.zero_grad()
                loss_m2.backward()
                optimizer_d2.step()

                optimizer_d3.zero_grad()
                loss_m3.backward()
                optimizer_d3.step()

                optimizer_d4.zero_grad()
                loss_m4.backward()
                optimizer_d4.step()
                

                optimizer_d5.zero_grad()
                loss_m5.backward()
                optimizer_d5.step()

            Mine_train_loss.append(running_mst/20)
            #print("JSD loss from training is: " + str(running_loss/30))
            print("JSD mst from training is: " + str(running_mst/20))


        # compute output
        output, target, q_l1, q_l2, q_l3, q_l4, backbone_feature = model(im_q=images[0], im_k=images[1], intermediate=False)
        #Mine.eval()
        #Mine3.eval()
        #Mine2.eval()
        
        if (using_discriminators):
            new_batch1, new_batch2, new_batch3, new_batch4, new_batch5 = compute_cat(q_l1, q_l2, q_l3, q_l4, backbone_feature, joint_q1.cuda(), joint_q2.cuda(), joint_q3.cuda(), joint_q4.cuda(), joint_q.cuda())
            
            loss_mine1, mst_mine1 = Discriminator1(new_batch1)
            loss_mine2, mst_mine2 = Discriminator2(new_batch2)
            loss_mine3, mst_mine3 = Discriminator3(new_batch3)
            loss_mine4, mst_mine4 = Discriminator4(new_batch4)
            
            loss_mine5, mst_mine5 = Global1(new_batch5)

            Mine_loss.append(mst_mine1.item()+mst_mine2.item()+mst_mine3.item()+mst_mine4.item()+mst_mine5.item())
            #Mine_loss.append(mst_mine5.item())
            print("JSD mst is: " + str(Mine_loss[-1]))

            loss = criterion(output, target)
            SL_loss.append(loss.item())
            print("SL_Moco loss is: " + str(loss))
            loss = loss - 0.5*loss_mine1 - 0.5*loss_mine2 - 0.5*loss_mine3 - 0.5*loss_mine4 - 0.5*loss_mine5
            #loss = loss - loss_mine5
        else:
            loss = criterion(output, target)
            SL_loss.append(loss.item())
            #print("SL_Moco loss is: " + str(loss))

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    if (using_discriminators):
        Mine_train_loss_epoch.append(sum(Mine_train_loss)/len(Mine_train_loss))
        np.save('JSD_train_mst.npy', Mine_train_loss_epoch)
        
        Mine_loss_epoch.append(sum(Mine_loss)/len(Mine_loss))
        np.save('JSD_mst.npy', Mine_loss_epoch)
    #np.save(str(epoch)+'_SL_loss.npy', SL_loss)
    Mine_train_loss_epoch = []
    Mine_loss_epoch = []
    SL_loss_epoch.append(sum(SL_loss)/len(SL_loss))
    np.save('SL_loss_fairdcl.npy', SL_loss_epoch)



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
