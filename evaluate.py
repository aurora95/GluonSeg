from __future__ import absolute_import

import os
import sys
import argparse
import mxnet as mx
import numpy as np
from PIL import Image
from mxnet import gluon, gpu
from datasets import *
from data_utils import DataTransformer, img_to_array, array_to_img
from models import *
from utils import *
from inference import inference

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation network')
    parser.add_argument('model_name',
                        help='the name of model definition function',
                        default=None, type=str)
    parser.add_argument('dataset',
                        help='dataset',
                        default=None, type=str)

    parser.add_argument('--crop_size', dest='crop_size', nargs=2, default=[512, 512], type=int)
    parser.add_argument('--ch_mean', dest='ch_mean', nargs=3, default=[0.485*255, 0.456*255, 0.406*255], type=float)
    parser.add_argument('--ch_std', dest='ch_std', nargs=3, default=[0.229*255, 0.224*255, 0.225*255], type=float)

    parser.add_argument('--gpu', dest='gpus',
                        nargs='*',
                        help='GPU device id to use',
                        default=[0], type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def calculate_iou(args, save_dir, image_list):
    train_file_path, val_file_path, data_dir, label_dir = get_dataset_path(args.dataset)
    classes = get_dataset_classes(args.dataset)
    image_suffix, label_suffix = get_dataset_suffix(args.dataset)

    conf_m = np.zeros((classes,classes), dtype=float)
    total = 0
    #mean_acc = 0.
    for img_name in image_list:
        img_name = img_name.strip('\n')
        total += 1
        print('#%d: %s'%(total,img_name))
        pred = img_to_array(Image.open('%s/%s.png'%(save_dir, img_name)), data_format='channels_last').astype(int)
        label = img_to_array(Image.open('%s/%s%s'%(label_dir, img_name, label_suffix)), data_format='channels_last').astype(int)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)
        #acc = 0.
        for p,l in zip(flat_pred,flat_label):
            if l==255:
                continue
            conf_m[l,p] += 1
        #    if l==p:
        #        acc+=1
        #acc /= flat_pred.shape[0]
        #mean_acc += acc
    #mean_acc /= total
    #print('mean acc: %f'%mean_acc)
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU

def evaluate(net, args):
    train_file_path, val_file_path, data_dir, label_dir = get_dataset_path(args.dataset)
    classes = get_dataset_classes(args.dataset)
    image_suffix, label_suffix = get_dataset_suffix(args.dataset)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = 'results/%s/'%args.dataset
    save_dir += '%s/res/'%args.model_name
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    fp = open(val_file_path)
    image_list = fp.readlines()
    fp.close()
    image_list = [img_name.strip('\n') for img_name in image_list]

    start_time = time.time()
    inference(net, args, image_list, return_results=False, save_dir=save_dir)
    duration = time.time() - start_time
    print('{}s used to make predictions.\n'.format(duration))

    start_time = time.time()
    conf_m, IOU, meanIOU = calculate_iou(args, save_dir, image_list)
    print('IOU: ')
    print(IOU)
    print('meanIOU: %f'%meanIOU)
    print('pixel acc: %f'%(np.sum(np.diag(conf_m))/np.sum(conf_m)))
    duration = time.time() - start_time
    print('{}s used to calculate IOU.\n'.format(duration))

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    net = globals()[model_name]
    evaluate(net, args)
