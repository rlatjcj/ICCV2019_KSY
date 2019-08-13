from __future__ import print_function

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from losses import TotalLoss
from model.retinanet import RetinaNet
from preprocessing import ListDataset
from utils import *

from torch.autograd import Variable

def create_generator(args):
    print('==> Preparing data..')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    trainset = ListDataset(root=args.root,
                           group=args.group,
                           list_file=args.trainset, 
                           train=True, 
                           transform=transform, 
                           input_size=args.input_size)

    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=True, 
                                              num_workers=8, 
                                              collate_fn=trainset.collate_fn)

    valset = ListDataset(root=args.root,
                         group=args.group,
                         list_file=args.valset, 
                         train=False, 
                         transform=transform, 
                         input_size=args.input_size,
                         shuffle=False)

    valloader = torch.utils.data.DataLoader(valset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             num_workers=8, 
                                             collate_fn=valset.collate_fn)
    
    return trainset, trainloader, valset, valloader

def create_model(args, classes):
    net = RetinaNet(backbone=args.backbone,
                    classes=classes)

    # net.load_state_dict(torch.load('./model/net.pth'))
    if args.checkpoint:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()

    return net


def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
                        
    parser.add_argument('--group', type=str,
                        required=True,
                        choices=['all', 'main', 'sub'],
                        help='Group used in training.')
    parser.add_argument('--root', 
                        required=True,
                        help='Path to image files.')
    parser.add_argument('--trainset', 
                        required=True,
                        help='Path to CSV file containing annotations for training.')
    parser.add_argument('--valset', 
                        required=False,
                        help='Path to CSV file containing annotations for validation (optional).')

    parser.add_argument('--backbone', type=str, 
                        required=True,
                        choices=['res50', 'res101', 'seres50', 'seres101'],
                        help='Backbone for retinanet.')
    parser.add_argument('--lr', default=.0001, type=float)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--steps', default=0, type=int)
    parser.add_argument('--val-steps', default=0, type=int)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--input-size', default=512, type=int)

    parser.add_argument('--checkpoint', default=None, type=str, help='Enter the checkpoint path if resuming training.')

    return parser.parse_args()


def main():
    args = get_arguments()
    assert args.command in ["train", "test"], 'Command must be choosen either \'train\' or \'test\'.'
    assert torch.cuda.is_available(), 'Error: CUDA not found!'

    ##############################################
    # Set Hyper Parameters
    ##############################################
    CLASSES = {'all': 599,
               'main': 70,
               'sub': 400} # will change

    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch

    # Data
    trainset, trainloader, valset, valloader = create_generator(args)

    # Model
    net = create_model(args, CLASSES[args.group])

    # Loss
    criterion = TotalLoss(classes=CLASSES[args.group])

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch {}/{}'.format(epoch+1, args.epochs))
        net.train()
        net.module.freeze_bn()
        train_loss = 0
        train_loc_loss = 0
        train_cls_loss = 0
        total_time = 0
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
            start = time.time()
            inputs = Variable(inputs.cuda())
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())

            optimizer.zero_grad()
            loc_preds, cls_preds = net(inputs)
            loss, loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loc_loss += loc_loss.data
            train_cls_loss += cls_loss.data
            duration = time.time() - start
            total_time += duration
            residual = cnt_time(args, trainset, duration, batch_idx)
            printProgress(batch_idx+1, args.steps if args.steps else len(trainset)//args.batch_size, 
                          '{}'.format(residual), 'loss: {:.4f} | loc_loss: {:.4f} | cls_loss: {:.4f}'.format(train_loss/(batch_idx+1),
                                                                                                             train_loc_loss/(batch_idx+1),
                                                                                                             train_cls_loss/(batch_idx+1)))

            if batch_idx+1 == args.steps and args.steps > 0:
                break

        print('\nValidation')
        net.eval()
        val_loss = 0
        val_loc_loss = 0
        val_cls_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(valloader):
                start = time.time()
                inputs = Variable(inputs.cuda())
                loc_targets = Variable(loc_targets.cuda())
                cls_targets = Variable(cls_targets.cuda())

                loc_preds, cls_preds = net(inputs)
                loss, loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
                val_loss += loss.data
                val_loc_loss += loc_loss.data
                val_cls_loss += cls_loss.data
                duration = time.time() - start
                total_time += duration
                residual = cnt_time(args, valset, duration, batch_idx)
                printProgress(batch_idx+1, args.val_steps if args.val_steps else len(valset)//args.batch_size, 
                            '{}'.format(residual), 'loss: {:.4f} | loc_loss: {:.4f} | cls_loss: {:.4f}'.format(val_loss/(batch_idx+1),
                                                                                                                val_loc_loss/(batch_idx+1),
                                                                                                                val_cls_loss/(batch_idx+1)))

                if batch_idx+1 == args.val_steps and args.val_steps > 0:
                    break

        # Save checkpoint
        # global best_loss
        # val_loss /= len(valloader)
        # if val_loss < best_loss:
        #     print('Saving..')
        #     state = {
        #         'net': net.module.state_dict(),
        #         'loss': val_loss,
        #         'epoch': epoch,
        #     }
        #     if not os.path.isdir('checkpoint'):
        #         os.mkdir('checkpoint')
        #     torch.save(state, './checkpoint/ckpt.pth')
        #     best_loss = val_loss

if __name__ == "__main__":
    main()