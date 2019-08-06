from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
from datagen import ListDataset

from torch.autograd import Variable

def create_generator():
    pass

def create_model():
    pass

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.module.freeze_bn()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss/(batch_idx+1)))


# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx+1)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = test_loss


def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
                        
    parser.add_argument('--group', type=str,
                        required=True,
                        choices=['all', 'main', 'sub'],
                        help='Group used in training.')
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
    parser.add_argument('--lr', default=.0001, type=float, 
                        help='Learning rate for training')
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='Optimizer for training')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Epochs for training')

    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Enter the checkpoint path if resuming training.')

    return parser.parse_args()


def main():
    args = get_arguments()
    assert args.command in ["train", "test"], 'Command must be choosen between \'train\' and \'test\'.'
    assert torch.cuda.is_available(), 'Error: CUDA not found!'

    ##############################################
    # Set Hyper Parameters
    ##############################################
    CLASSES = {'all': 600,
               'main': 70,
               'sub': 400} # will change

    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch

    # Data
    print('==> Preparing data..')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    trainset = ListDataset(root=args.root,
                           list_file=args.trainset, 
                           train=True, 
                           transform=transform, 
                           input_size=600)

    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=16, 
                                              shuffle=True, 
                                              num_workers=8, 
                                              collate_fn=trainset.collate_fn)

    valset = ListDataset(root=args.root,
                         list_file=args.valset, 
                         train=False, 
                         transform=transform, 
                         input_size=600)

    testloader = torch.utils.data.DataLoader(valset, 
                                             batch_size=16, 
                                             shuffle=False, 
                                             num_workers=8, 
                                             collate_fn=valset.collate_fn)

    # Model
    net = RetinaNet(backbone=args.backbone,
                    classes=CLASSES[args.group])

    net.load_state_dict(torch.load('./model/net.pth'))
    if args.checkpoint:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()

    criterion = FocalLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(start_epoch, args.epochs):
        train(epoch)
        test(epoch)

if __name__ == "__main__":
    main()