from __future__ import division
import os.path
from .listdataset import ListDataset

import numpy as np
import flow_transforms

try:
    import cv2
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load openCV, which is needed"
                      "for KITTI which uses 16bit PNG images", ImportWarning)

'''
Data load for bsds500 dataset:
author: Fengting Yang 
Mar.1st 2019

usage:
1. manually change the name of train.txt and val.txt in the make_dataset(dir) func.    
2. ensure the val_dataset using the same size as the args. in the main code when performing centerCrop 
   default value is 320*320, it is fixed to be 16*n in our project
'''


def make_dataset_list(root_dir):
    # we train and val separately to tune the hyper-param
    # use all the data for the final training
    train_list_dir = os.path.join(root_dir, 'train.txt')
    # use train_Val.txt for final report
    val_list_dir = os.path.join(root_dir, 'val.txt')

    try:
        with open(train_list_dir, 'r') as tf:
            train_list = tf.readlines()
        with open(val_list_dir, 'r') as vf:
            val_list = vf.readlines()

    except IOError:
        print('Error! No available list.')
        return

    return train_list, val_list


def BSD_loader(img_path, label):
    # cv2.imread is faster than io.imread usually
    img = cv2.imread(img_path)[:, :, ::-1].astype(np.float32)
    gt_seg = cv2.imread(label)[:, :, :1]
    return img, gt_seg


def BSD500(root, train_transform=None, label_transform=None, val_transform=None, co_transform=None, split=None):
    train_list, val_list = make_dataset_list(root)

    if val_transform is None:
        val_transform = train_transform

    train_dataset = ListDataset(root, 'bsd500', train_list, train_transform,
                                label_transform, co_transform,
                                loader=BSD_loader, datatype='train')

    val_dataset = ListDataset(root, 'bsd500', val_list, val_transform,
                              label_transform, flow_transforms.CenterCrop((320, 320)),
                              loader=BSD_loader, datatype='val')

    return train_dataset, val_dataset


