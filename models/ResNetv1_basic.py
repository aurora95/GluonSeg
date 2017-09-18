from __future__ import absolute_import

import mxnet as mx
from models.base_models import *
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn

class ResNet50_16s(HybridBlock):
    def __init__(self, classes, **kwargs):
        super(ResNet50_16s, self).__init__(**kwargs)
        with self.name_scope():
            self.classes = classes
            self.resnet = get_resnet(1, num_layers=50, pretrained=False, remove_subsample=1)

            self.classifier = nn.Conv2D(classes, kernel_size=(1, 1), weight_initializer=mx.init.Xavier())
            #self.classifier.collect_params().setattr('lr_mult', 10)

            self.upsampler = nn.Conv2DTranspose(channels=classes, kernel_size=32, strides=16, padding=8,
                                                groups=classes, use_bias=False, weight_initializer=mx.init.Bilinear())
            self.upsampler.collect_params().setattr('lr_mult', 0.)

    def hybrid_forward(self, F, x):
        features = self.resnet(x)
        y = self.classifier(features[len(features)-1])
        y = self.upsampler(y)
        #print(self.resnet.module3.collect_params().get('conv7_weight').data())
        return y

    def load_base_model(self, ctx):
        from mxnet.gluon.model_zoo.model_store import get_model_file
        self.resnet.load_params(get_model_file('resnet50_v1'), ctx=ctx, allow_missing=False, ignore_extra=True)
