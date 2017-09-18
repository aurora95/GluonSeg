# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
"""ResNets, implemented in Gluon."""
from __future__ import division

__all__ = ['ResNetV1',
           'BasicBlockV1',
           'BottleneckV1',
           'get_resnet']

from mxnet.base import numeric_types
from mxnet.context import cpu
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn

# Helpers
def _conv3x3(channels, stride, in_channels, dilation=(1, 1)):
    if isinstance(dilation, numeric_types):
        dilation = (dilation,)*len(kernel_size)
    padding = dilation[0]
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=padding,
                     dilation=dilation, use_bias=False, in_channels=in_channels)


# Blocks
class BasicBlockV1(HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, dilation=(1, 1), **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels, dilation=dilation))
        self.body.add(nn.BatchNorm(momentum=0.99))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels, dilation=dilation))
        self.body.add(nn.BatchNorm(momentum=0.99))
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm(momentum=0.99))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual+x, act_type='relu')

        return x


class BottleneckV1(HybridBlock):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, dilation=(1, 1), **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm(momentum=0.99))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels//4, stride, channels//4, dilation=dilation))
        self.body.add(nn.BatchNorm(momentum=0.99))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm(momentum=0.99))
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm(momentum=0.99))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x

# Nets
class ResNetV1(HybridBlock):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    thumbnail : bool, default False
        Enable thumbnail.
    """
    def __init__(self, block, layers, channels, remove_subsample=0, thumbnail=False, **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        self.num_modules = len(layers)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.entry = nn.HybridSequential(prefix='')
            if thumbnail:
                self.entry.add(_conv3x3(channels[0], 1, 3))
            else:
                self.entry.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False,
                                            in_channels=3))
                self.entry.add(nn.BatchNorm(momentum=0.99))
                self.entry.add(nn.Activation('relu'))
                self.entry.add(nn.MaxPool2D(3, 2, 1))

            dilation = 1
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                if i >= self.num_modules - remove_subsample:
                    stride = 1
                    dilation *= 2
                self.__setattr__('module%d'%i, self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i],
                                                   dilation=(dilation, dilation)))

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0, dilation=(1, 1)):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            dilation=dilation, prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, dilation=dilation, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.entry(x)
        features = []
        for i in range(self.num_modules):
            x = self.__dict__['module%d'%i](x)
            features.append(x)

        return features


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1]#, ResNetV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1}]


# Constructor
def get_resnet(version, num_layers, ctx=None, pretrained=False, remove_subsample=0, **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    block_type, layers, channels = resnet_spec[num_layers]
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, remove_subsample, **kwargs)
    if pretrained:
        from mxnet.gluon.model_zoo.model_store import get_model_file
        net.load_params(get_model_file('resnet%d_v%d'%(num_layers, version)), ctx=ctx, allow_missing=False, ignore_extra=True)
    return net
