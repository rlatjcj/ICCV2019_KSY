'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import json
import tqdm
import random
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from .encoder import DataEncoder
from .transform import resize, random_flip, random_crop, center_crop


class ListDataset(data.Dataset):
    def __init__(self, 
                 root,
                 group,
                 list_file, 
                 train, 
                 transform, 
                 input_size,
                 shuffle=True):
        '''
        Args:
            root: (str) ditectory to images.
            list_file: (str) path to index file.
            train: (boolean) train or test.
            transform: ([transforms]) image transforms.
            input_size: (int) model input size.
        '''
        self.root = root
        self.group = group
        self.train = train
        self.transform = transform
        self.input_size = input_size
        self.shuffle = shuffle

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()

        self.classid = json.loads(open("./data/classid_{}.json".format(group), "r").read())
        self.df = pd.read_csv(list_file, header=None)
        self.fnames = np.unique(self.df[0].values).tolist()            

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        if self.shuffle and idx == 0:
            random.shuffle(self.fnames)

        # Load image and boxes.
        fname = self.fnames[idx]
        try:
            img = Image.open(os.path.join(self.root, fname))
        except:
            img = Image.open(os.path.join(self.root, fname.replace(fname.split('/')[0], 'train')))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxlist = self.df[self.df[0] == fname]
        boxes = torch.Tensor([[float(boxlist.iloc[i,1]), float(boxlist.iloc[i,2]), float(boxlist.iloc[i,3]), float(boxlist.iloc[i,4])] for i in range(len(boxlist))])
        labels = torch.LongTensor([int(self.classid[boxlist.iloc[i,5]]) for i in range(len(boxlist))])
        size = self.input_size

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size,size))

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return len(self.fnames)


if __name__ == "__main__":
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    dataset = ListDataset(root='/data/public/rw/kiminhwan/jshp',
                          group='all',
                          list_file='./data/validation_for_retina.csv', 
                          train=True, 
                          transform=transform, 
                          input_size=600)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    for i, (images, loc_targets, cls_targets) in enumerate(dataloader):
        print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        # print('loc_targets :', loc_targets)
        print(i, 'cls_targets :', cls_targets.unique())
        break
        # grid = torchvision.utils.make_grid(images, 1)
        # torchvision.utils.save_image(grid, './test.jpg')