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

    parser.add_argument('--image_names', dest='image_names',
                        nargs='*',
                        default=[], type=str)
    parser.add_argument('--gpu', dest='gpus',
                        nargs='*',
                        help='GPU device id to use',
                        default=[0], type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def inference(net, args, image_names, return_results=True, save_dir=None):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = 'results/%s/'%args.dataset
    save_path += '%s/'%args.model_name

    ctx = [gpu(i) for i in args.gpus]
    train_file_path, val_file_path, data_dir, label_dir = get_dataset_path(args.dataset)
    image_suffix, label_suffix = get_dataset_suffix(args.dataset)
    classes = get_dataset_classes(args.dataset)
    net = net(classes)
    net.collect_params().initialize(ctx=ctx)
    net.load_params(save_path+'checkpoint.params', ctx)

    results = []
    total = 0
    for name in image_names:
        total += 1
        print('#%d: %s'%(total,name))
        img_file = os.path.join(data_dir, name + image_suffix)
        label_file = os.path.join(label_dir, name + label_suffix)
        image = Image.open(img_file)
        label = Image.open(label_file)
        img_h, img_w = img_to_array(image, data_format='channels_last').shape[0:2]
        pad_w = max(args.crop_size[1] - img_w, 0)
        pad_h = max(args.crop_size[0] - img_h, 0)

        transformer = DataTransformer(ch_mean=args.ch_mean, ch_std=args.ch_std, crop_mode='center',
                                      crop_size=args.crop_size, data_format='channels_first')
        image, _ = transformer.transform(image, label)
        image = mx.nd.array(np.expand_dims(image, axis=0)).as_in_context(ctx[0])
        output = net(image)

        pred_label = mx.nd.argmax(output, axis=1, keepdims=False).asnumpy()
        pred_label = np.squeeze(pred_label).astype(np.uint8)
        result_img = Image.fromarray(pred_label, mode='P')
        result_img = result_img.crop((pad_w//2, pad_h//2, pad_w//2+img_w, pad_h//2+img_h))
        result_img.putpalette(label.getpalette())
        if return_results:
            results.append(result_img)
        if save_dir:
            result_img.save(os.path.join(save_dir, name + '.png'))
    return results


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    net = globals()[model_name]
    results = inference(net, args, args.image_names, True)
    for result in results:
        result.show(title='result', command=None)
