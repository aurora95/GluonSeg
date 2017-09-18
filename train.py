from __future__ import absolute_import

import os
import sys
import argparse
import mxnet as mx
import numpy as np
from mxnet import gluon, gpu
from datasets import *
from data_utils import GeneratorEnqueuer, DataTransformer
from models import *
from utils import *
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation network')
    parser.add_argument('model_name',
                        help='the name of model definition function',
                        default=None, type=str)
    parser.add_argument('dataset', default=None, type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=16, type=int)
    parser.add_argument('--epochs', dest='epochs', default=30, type=int)

    parser.add_argument('--base_lr', dest='base_lr', default=0.01, type=float)
    parser.add_argument('--lr_power', dest='lr_power', default=0.9, type=float)
    parser.add_argument('--resize_size', dest='resize_size', nargs=2, default=None, type=int)
    parser.add_argument('--pad_size', dest='pad_size', nargs=2, default=None, type=int)
    parser.add_argument('--crop_size', dest='crop_size', nargs=2, default=[480, 480], type=int)
    parser.add_argument('--crop_mode', dest='crop_mode', default='random', type=str)
    parser.add_argument('--ch_mean', dest='ch_mean', nargs=3, default=[0.485*255, 0.456*255, 0.406*255], type=float)
    parser.add_argument('--ch_std', dest='ch_std', nargs=3, default=[0.229*255, 0.224*255, 0.225*255], type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.0001, type=float)

    parser.add_argument('--workers', dest='workers', default=4, type=int)
    parser.add_argument('--max_queue_size', dest='max_queue_size', default=16, type=int)
    parser.add_argument('--gpu', dest='gpus',
                        nargs='*',
                        help='GPU device id to use',
                        default=[0], type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def train_batch(batch_x, batch_y, ctx, net, trainer, loss, metrics):
    # split the data batch and load them on GPUs
    data = gluon.utils.split_and_load(batch_x, ctx)
    label = gluon.utils.split_and_load(batch_y, ctx)
    # compute gradient
    with gluon.autograd.record():
        preds = [net(X) for X in data]
        losses = [loss(pred, Y) for pred, Y in zip(preds, label)]
    for l in losses:
        l.backward()
    # update parameters
    trainer.step(len(data))
    for m in metrics:
        m.update(labels=label, preds=preds)
    return losses

def train(net, args):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(os.path.join(current_dir, 'results/')) == False:
        os.mkdir(os.path.join(current_dir, 'results/'))
    save_path = 'results/%s/'%args.dataset
    if os.path.exists(os.path.join(current_dir, save_path)) == False:
        os.mkdir(os.path.join(current_dir, save_path))
    save_path += '%s/'%args.model_name
    if os.path.exists(os.path.join(current_dir, save_path)) == False:
        os.mkdir(os.path.join(current_dir, save_path))
    logger = Logger(save_path + 'logs/')

    train_file_path, val_file_path, data_dir, label_dir = get_dataset_path(args.dataset)
    classes = get_dataset_classes(args.dataset)
    transformer = DataTransformer(ch_mean=args.ch_mean, ch_std=args.ch_std, resize_size=args.resize_size,
                 pad_size=args.pad_size, crop_mode=args.crop_mode, crop_size=args.crop_size,
                 zoom_range=[0.5, 2.0], horizontal_flip=True, color_jittering_range=20.,
                 fill_mode='constant', cval=0., label_cval=255, data_format='channels_first',
                 color_format='RGB', x_dtype=np.float32)
    dataloader = VOC12(data_list_file=train_file_path, data_source_dir=data_dir,
                       label_source_dir=label_dir, data_transformer=transformer,
                       batch_size=args.batch_size, shuffle=True)

    ctx = [gpu(i) for i in args.gpus]
    net = net(classes)
    net.collect_params().initialize(ctx=ctx)
    net.load_base_model(ctx)
    #net.hybridize()
    #print(net)

    num_sample = dataloader.get_num_sample()
    num_steps = num_sample//args.batch_size
    if num_sample % args.batch_size > 0:
        num_steps += 1

    enqueuer = GeneratorEnqueuer(generator=dataloader)
    enqueuer.start(workers=args.workers, max_queue_size=args.max_queue_size)
    output_generator = enqueuer.get()

    trainer = gluon.Trainer(net.collect_params(), 'nag',
                            {'momentum': 0.9, 'wd': 0.0001,
                            'learning_rate': args.base_lr,
                            'lr_scheduler': PolyScheduler(args.base_lr, args.lr_power, num_steps*args.epochs)})
    loss = MySoftmaxCrossEntropyLoss(axis=1, ignore_label=255)
    metrics = [AccuracyWithIgnoredLabel(axis=1, ignore_label=255)]

    for epoch in range(args.epochs):
        print('training epoch %d/%d:'%(epoch+1, args.epochs))
        for m in metrics:
            m.reset()
        train_loss = 0.
        train_acc = 0.
        for i in range(num_steps):
            batch_x, batch_y = next(output_generator)

            batch_x = mx.nd.array(batch_x)
            batch_y = mx.nd.array(batch_y)

            losses = train_batch(batch_x, batch_y, ctx, net, trainer, loss, metrics)

            train_loss += mx.nd.mean(mx.nd.add_n(*losses)).asscalar()/len(args.gpus)
            info = 'loss: %.3f' % (train_loss/(i+1))
            for m in metrics:
                name, value = m.get()
                info += ' | %s: %.3f'%(name, value)
            progress_bar(i, num_steps, info)
        # write logs for this epoch
        logger.scalar_summary('loss', train_loss/num_steps, epoch)
        for m in metrics:
            name, value = m.get()
            logger.scalar_summary(name, value, epoch)
        mx.nd.waitall()
        net.save_params(save_path+'checkpoint.params')

    enqueuer.stop()

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    net = globals()[model_name]
    train(net, args)
