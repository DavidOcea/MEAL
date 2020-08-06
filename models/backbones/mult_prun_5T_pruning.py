from functools import partial
import json
import math
import os
import sys
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import torch
import torch.nn as nn
import torch.optim
from collections import OrderedDict
import copy

__all__ = ['mult_prun8_gpu_prun']


model_urls = {
	'proxyless_cpu': 'https://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.pth',
	'proxyless_gpu': 'https://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.pth',
	'proxyless_mobile': 'https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.pth',
	'proxyless_mobile_14': 'https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile_14.pth',
    'multnas_gpu':'qiniu:///personal/yangdecheng/pretrained_model/multnas_gpu.pth.tar',
}

gpu_config = {
	'proxyless_cpu': 'https://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.pth',
	'proxyless_gpu': 'https://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.pth',
	'proxyless_mobile': 'https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.pth',
	'proxyless_mobile_14': 'https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile_14.pth',
    'multnas_gpu':'qiniu:///personal/yangdecheng/pretrained_model/multnas_gpu.pth.tar',
}

gpu_config = {
 "name": "ProxylessNASNets",
 "bn": {
  "momentum": 0.1,
  "eps": 0.001
 },
 "first_conv": {
  "name": "ConvLayer",
  "kernel_size": 3,
  "stride": 2,
  "dilation": 1,
  "groups": 1,
  "bias": False,
  "has_shuffle": False,
  "in_channels": 3,
  "out_channels": 64,
  "use_bn": True,
  "act_func": "relu6",
  "dropout_rate": 0,
  "ops_order": "weight_bn_act"
 },
 "blocks": [
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 64,
    "out_channels": 32,
    "kernel_size": 3,
    "stride": 1,
    "expand_ratio": 1,
    "flag": False,
    "inverted_out": 64,
    "depth_out": 64,
    "flag_channels": True
   },
   "shortcut": None
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 32,
    "out_channels": 48,
    "kernel_size": 5,
    "stride": 2,
    "expand_ratio": 6,
    "flag": False,
    "inverted_out": 66,
    "depth_out": 66,
    "flag_channels": True
   },
   "shortcut": None
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "ZeroLayer",
    "stride": 1
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 48,
    "out_channels": 48,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 48,
    "out_channels": 48,
    "kernel_size": 5,
    "stride": 1,
    "expand_ratio": 6,
    "flag": True,
    "inverted_out": 34,
    "depth_out": 34,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 48,
    "out_channels": 48,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "ZeroLayer",
    "stride": 1
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 48,
    "out_channels": 48,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 48,
    "out_channels": 80,
    "kernel_size": 3,
    "stride": 2,
    "expand_ratio": 3,
    "flag": False,
    "inverted_out": 58,
    "depth_out": 58,
    "flag_channels": True
   },
   "shortcut": None
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 80,
    "out_channels": 80,
    "kernel_size": 7,
    "stride": 1,
    "expand_ratio": 6,
    "flag": True,
    "inverted_out": 38,
    "depth_out": 38,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 80,
    "out_channels": 80,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 80,
    "out_channels": 80,
    "kernel_size": 7,
    "stride": 1,
    "expand_ratio": 6,
    "flag": True,
    "inverted_out": 34,
    "depth_out": 34,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 80,
    "out_channels": 80,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 80,
    "out_channels": 80,
    "kernel_size": 7,
    "stride": 1,
    "expand_ratio": 3,
    "flag": True,
    "inverted_out": 20,
    "depth_out": 20,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 80,
    "out_channels": 80,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 80,
    "out_channels": 160,
    "kernel_size": 5,
    "stride": 2,
    "expand_ratio": 6,
    "flag": False,
    "inverted_out": 140,
    "depth_out": 140,
    "flag_channels": True
   },
   "shortcut": None
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 160,
    "out_channels": 160,
    "kernel_size": 7,
    "stride": 1,
    "expand_ratio": 6,
    "flag": True,
    "inverted_out": 44,
    "depth_out": 44,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 160,
    "out_channels": 160,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 160,
    "out_channels": 160,
    "kernel_size": 7,
    "stride": 1,
    "expand_ratio": 6,
    "flag": True,
    "inverted_out": 42,
    "depth_out": 42,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 160,
    "out_channels": 160,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 160,
    "out_channels": 160,
    "kernel_size": 5,
    "stride": 1,
    "expand_ratio": 6,
    "flag": True,
    "inverted_out": 22,
    "depth_out": 22,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 160,
    "out_channels": 160,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 160,
    "out_channels": 192,
    "kernel_size": 5,
    "stride": 1,
    "expand_ratio": 6,
    "flag": False,
    "inverted_out": 442,
    "depth_out": 442,
    "flag_channels": True
   },
   "shortcut": None
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 192,
    "out_channels": 192,
    "kernel_size": 5,
    "stride": 1,
    "expand_ratio": 3,
    "flag": True,
    "inverted_out": 10,
    "depth_out": 10,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 192,
    "out_channels": 192,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 192,
    "out_channels": 192,
    "kernel_size": 5,
    "stride": 1,
    "expand_ratio": 6,
    "flag": True,
    "inverted_out": 6,
    "depth_out": 6,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 192,
    "out_channels": 192,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 192,
    "out_channels": 192,
    "kernel_size": 7,
    "stride": 1,
    "expand_ratio": 6,
    "flag": True,
    "inverted_out": 16,
    "depth_out": 16,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 192,
    "out_channels": 192,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 192,
    "out_channels": 384,
    "kernel_size": 7,
    "stride": 2,
    "expand_ratio": 6,
    "flag": False,
    "inverted_out": 62,
    "depth_out": 62,
    "flag_channels": True
   },
   "shortcut": None
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 384,
    "out_channels": 384,
    "kernel_size": 5,
    "stride": 1,
    "expand_ratio": 6,
    "flag": True,
    "inverted_out": 2,
    "depth_out": 2,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 384,
    "out_channels": 384,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 384,
    "out_channels": 384,
    "kernel_size": 5,
    "stride": 1,
    "expand_ratio": 6,
    "flag": True,
    "inverted_out": 6,
    "depth_out": 6,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 384,
    "out_channels": 384,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 384,
    "out_channels": 384,
    "kernel_size": 7,
    "stride": 1,
    "expand_ratio": 3,
    "flag": True,
    "inverted_out": 6,
    "depth_out": 6,
    "flag_channels": True
   },
   "shortcut": {
    "name": "IdentityLayer",
    "in_channels": 384,
    "out_channels": 384,
    "use_bn": False,
    "act_func": None,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
   }
  },
  {
   "name": "MobileInvertedResidualBlock",
   "mobile_inverted_conv": {
    "name": "MBInvertedConvLayer",
    "in_channels": 384,
    "inverted_out": 20,
    "depth_out": 20,
    "out_channels": 640,
    "kernel_size": 5,
    "stride": 1,
    "expand_ratio": 6,
    "flag": False,
    "flag_channels": True
   },
   "shortcut": None
  }
 ],
 "feature_mix_layer": {
  "name": "ConvLayer",
  "kernel_size": 1,
  "stride": 1,
  "dilation": 1,
  "groups": 1,
  "bias": False,
  "has_shuffle": False,
  "in_channels": 640,
  "out_channels": 18,
  "use_bn": True,
  "act_func": "relu6",
  "dropout_rate": 0,
  "ops_order": "weight_bn_act"
 },
 "classifier": {
  "use_bn": False,
  "ops_order": "weight_bn_act",
  "out_features": 9,
  "dropout_rate": 0,
  "bias": True,
  "act_func": None,
  "name": "LinearLayer",
  "in_features": 1280
 }
}


class GateLayer(nn.Module):
    def __init__(self, input_features, output_features, size_mask):
        super(GateLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.size_mask = size_mask
        self.weight = nn.Parameter(torch.ones(output_features))

        # for simpler way to find these layers
        self.do_not_update = True

    def forward(self, input):
        return input*self.weight.view(*self.size_mask)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.output_features is not None
        )

def download_url(url, model_dir="~/.torch/proxyless_nas", overwrite=False):
    model_dir = os.path.expanduser(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) or overwrite:
        os.makedirs(model_dir, exist_ok=True)
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return cached_file

class BasicUnit(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(
        kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2

def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        DepthConvLayer.__name__: DepthConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
        ZeroLayer.__name__: ZeroLayer,
    }
    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class BasicLayer(BasicUnit):

    def __init__(
            self,
            in_channels,
            out_channels,
            use_bn=True,
            act_func='relu',
            dropout_rate=0,
            ops_order='weight_bn_act'):
        super(BasicLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ add modules """
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                self.bn = nn.BatchNorm2d(in_channels)
            else:
                self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        # activation
        if act_func == 'relu':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU(inplace=False)
            else:
                self.activation = nn.ReLU(inplace=True)
        elif act_func == 'relu6':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU6(inplace=False)
            else:
                self.activation = nn.ReLU6(inplace=True)
        else:
            self.activation = None
        # dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            self.dropout = None

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_call(self, x):
        raise NotImplementedError

    def forward(self, x):
        for op in self.ops_list:
            if op == 'weight':
                # dropout before weight operation
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.weight_call(x)
            elif op == 'bn':
                if self.bn is not None:
                    x = self.bn(x)
            elif op == 'act':
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise ValueError('Unrecognized op: %s' % op)
        return x

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


class ConvLayer(BasicLayer):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            bias=False,
            has_shuffle=False,
            use_bn=True,
            act_func='relu',
            dropout_rate=0,
            ops_order='weight_bn_act'):
        super(
            ConvLayer,
            self).__init__(
            in_channels,
            out_channels,
            use_bn,
            act_func,
            dropout_rate,
            ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation
        # `kernel_size`, `stride`, `padding`, `dilation` can either be `int` or `tuple` of int
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias)

    def weight_call(self, x):
        x = self.conv(x)
        if self.has_shuffle and self.groups > 1:
            x = shuffle_layer(x, self.groups)
        return x

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                return '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                return '%dx%d_DilatedGroupConv' % (
                    kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        config = {
            'name': ConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
        }
        config.update(super(ConvLayer, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)

    def get_flops(self, x):
        return count_conv_flop(self.conv, x), self.forward(x)


class DepthConvLayer(BasicLayer):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            bias=False,
            has_shuffle=False,
            use_bn=True,
            act_func='relu',
            dropout_rate=0,
            ops_order='weight_bn_act'):
        super(
            DepthConvLayer,
            self).__init__(
            in_channels,
            out_channels,
            use_bn,
            act_func,
            dropout_rate,
            ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation
        # `kernel_size`, `stride`, `padding`, `dilation` can either be `int` or `tuple` of int
        self.depth_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=in_channels,
            bias=False)
        self.point_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=self.groups,
            bias=self.bias)

    def weight_call(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        if self.has_shuffle and self.groups > 1:
            x = shuffle_layer(x, self.groups)
        return x

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.dilation > 1:
            return '%dx%d_DilatedDepthConv' % (kernel_size[0], kernel_size[1])
        else:
            return '%dx%d_DepthConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        config = {
            'name': DepthConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
        }
        config.update(super(DepthConvLayer, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)

    def get_flops(self, x):
        depth_flop = count_conv_flop(self.depth_conv, x)
        point_flop = count_conv_flop(self.point_conv, self.depth_conv(x))
        return depth_flop + point_flop, self.forward(x)


class PoolingLayer(BasicLayer):

    def __init__(
            self,
            in_channels,
            out_channels,
            pool_type,
            kernel_size=2,
            stride=2,
            use_bn=False,
            act_func=None,
            dropout_rate=0,
            ops_order='weight_bn_act'):
        super(
            PoolingLayer,
            self).__init__(
            in_channels,
            out_channels,
            use_bn,
            act_func,
            dropout_rate,
            ops_order)

        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        if self.stride == 1:
            # same padding if `stride == 1`
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        if self.pool_type == 'avg':
            self.pool = nn.AvgPool2d(
                self.kernel_size,
                stride=self.stride,
                padding=padding,
                count_include_pad=False)
        elif self.pool_type == 'max':
            self.pool = nn.MaxPool2d(
                self.kernel_size,
                stride=self.stride,
                padding=padding)
        else:
            raise NotImplementedError

    def weight_call(self, x):
        return self.pool(x)

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return '%dx%d_%sPool' % (
            kernel_size[0], kernel_size[1], self.pool_type.upper())

    @property
    def config(self):
        config = {
            'name': PoolingLayer.__name__,
            'pool_type': self.pool_type,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
        }
        config.update(super(PoolingLayer, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class IdentityLayer(BasicLayer):

    def __init__(
            self,
            in_channels,
            out_channels,
            use_bn=False,
            act_func=None,
            dropout_rate=0,
            ops_order='weight_bn_act'):
        super(
            IdentityLayer,
            self).__init__(
            in_channels,
            out_channels,
            use_bn,
            act_func,
            dropout_rate,
            ops_order)

    def weight_call(self, x):
        return x

    @property
    def unit_str(self):
        return 'Identity'

    @property
    def config(self):
        config = {
            'name': IdentityLayer.__name__,
        }
        config.update(super(IdentityLayer, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class LinearLayer(BasicUnit):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            use_bn=False,
            act_func=None,
            dropout_rate=0,
            ops_order='weight_bn_act'):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ add modules """
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                self.bn = nn.BatchNorm1d(in_features)
            else:
                self.bn = nn.BatchNorm1d(out_features)
        else:
            self.bn = None
        # activation
        if act_func == 'relu':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU(inplace=False)
            else:
                self.activation = nn.ReLU(inplace=True)
        elif act_func == 'relu6':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU6(inplace=False)
            else:
                self.activation = nn.ReLU6(inplace=True)
        elif act_func == 'tanh':
            self.activation = nn.Tanh()
        elif act_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
        # dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        # linear
        self.linear = nn.Linear(self.in_features, self.out_features, self.bias)

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def forward(self, x):
        for op in self.ops_list:
            if op == 'weight':
                # dropout before weight operation
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.linear(x)
            elif op == 'bn':
                if self.bn is not None:
                    x = self.bn(x)
            elif op == 'act':
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise ValueError('Unrecognized op: %s' % op)
        return x

    @property
    def unit_str(self):
        return '%dx%d_Linear' % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            'name': LinearLayer.__name__,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)

    def get_flops(self, x):
        return self.linear.weight.numel(), self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False


class MBInvertedConvLayer(BasicUnit):

    def __init__(
            self,
            in_channels,
            inverted_out,
            depth_out,
            out_channels,
            kernel_size=3,
            stride=1,
            expand_ratio=6,
            flag=False,
            flag_channels=False):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.inverted_out = inverted_out
        self.depth_out = depth_out
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.flag = flag
        self.flag_channels =flag_channels

        
        if self.flag_channels:
            if self.expand_ratio > 1:
                self.inverted_bottleneck = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, inverted_out, 1, 1, 0, bias=False)),
                    ('bn', nn.BatchNorm2d(inverted_out)),
                    ('relu', nn.ReLU6(inplace=True)),
                ]))
            else:
                feature_dim = in_channels
                self.inverted_bottleneck = None
            
            # depthwise convolution
            pad = get_same_padding(self.kernel_size)
            self.depth_conv = nn.Sequential(
                OrderedDict(
                    [
                        ('conv',
                        nn.Conv2d(
                            inverted_out,
                            depth_out,
                            kernel_size,
                            stride,
                            pad,
                            groups=inverted_out,
                            bias=False)),
                        ('bn',
                        nn.BatchNorm2d(depth_out)),
                        ('relu',
                        nn.ReLU6(
                            inplace=True)),
                    ]))

            # pointwise linear
            self.point_linear = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(depth_out, out_channels, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels)),
            ]))
        else:
            if self.expand_ratio > 1:
                feature_dim = round(in_channels * self.expand_ratio)
                self.inverted_bottleneck = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, feature_dim, 1, 1, 0, bias=False)),
                    ('bn', nn.BatchNorm2d(feature_dim)),
                    ('gate', GateLayer(feature_dim,feature_dim,[1, -1, 1, 1])),
                    ('relu', nn.ReLU6(inplace=True)),
                ]))
            else:
                feature_dim = in_channels
                self.inverted_bottleneck = None
            
            # depthwise convolution
            pad = get_same_padding(self.kernel_size)            
            self.depth_conv = nn.Sequential(
                OrderedDict(
                    [
                        ('conv',
                            nn.Conv2d(
                                feature_dim,
                                feature_dim,
                                kernel_size,
                                stride,
                                pad,
                                groups=feature_dim,
                                bias=False)),
                            ('bn',
                            nn.BatchNorm2d(feature_dim)),
                            ('gate', GateLayer(feature_dim,feature_dim,[1, -1, 1, 1])),
                            ('relu',
                            nn.ReLU6(
                                inplace=True)),
                    ]))

            # pointwise linear
            self.point_linear = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels)),
            ]))

        

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        # print('MB_resule_inverted:',x)
        x = self.depth_conv(x)
        # print('MB_resule_depth:',x)
        x = self.point_linear(x)
        # print('MB_resule_point:',x)
        return x

    @property
    def unit_str(self):
        unit_str = '%dx%d_MBConv%d' % (
            self.kernel_size, self.kernel_size, self.expand_ratio)
        return unit_str

    @property
    def config(self):
        return {
            'name': MBInvertedConvLayer.__name__,
            'in_channels': self.in_channels,
            'inverted_out': self.inverted_out,
            'depth_out': self.depth_out,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'flag': self.flag,
            'flag_channels': self.flag_channels
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

    def get_flops(self, x):
        if self.inverted_bottleneck:
            flop1 = count_conv_flop(self.inverted_bottleneck.conv, x)
            x = self.inverted_bottleneck(x)
        else:
            flop1 = 0

        flop2 = count_conv_flop(self.depth_conv.conv, x)
        x = self.depth_conv(x)

        flop3 = count_conv_flop(self.point_linear.conv, x)
        x = self.point_linear(x)
        return flop1 + flop2 + flop3, x

    @staticmethod
    def is_zero_layer():
        return False


class ZeroLayer(BasicUnit):
    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        if x.is_cuda:
            with torch.cuda.device(x.get_device()):
                padding = torch.cuda.FloatTensor(n, c, h, w).fill_(0)
        else:
            padding = torch.zeros(n, c, h, w)
        padding = torch.autograd.Variable(padding, requires_grad=False)
        return padding

    @property
    def unit_str(self):
        return 'Zero'

    @property
    def config(self):
        return {
            'name': ZeroLayer.__name__,
            'stride': self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return True


class MobileInvertedResidualBlock(BasicUnit):
    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()
        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def unit_str(self):
        return '(%s, %s)' % (self.mobile_inverted_conv.unit_str,
                             self.shortcut.unit_str if self.shortcut is not None else None)

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(
            config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, _ = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)


class ProxylessNASNets(BasicUnit):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNets, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1) #cx 分支 空间注意力
        # self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.feature_mix_layer:
            x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)          #cx 分支 空间注意力 要注释掉
        x = x.view(x.size(0), -1)  # flatten    #cx 分支 空间注意力
        # x = self.classifier(x)
        return x

    @property
    def unit_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'feature_mix_layer': self.feature_mix_layer.config if self.feature_mix_layer is not None else None,
            'classifier': self.classifier.config,
            'blocks': [
                block.config for block in self.blocks],
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])
        blocks = []
        for block_config in config['blocks']:
            blocks.append(
                MobileInvertedResidualBlock.build_from_config(block_config))

        return ProxylessNASNets(
            first_conv,
            blocks,
            feature_mix_layer,
            classifier)

    def get_flops(self, x):
        flop, x = self.first_conv.get_flops(x)

        for block in self.blocks:
            delta_flop, x = block.get_flops(x)
            flop += delta_flop
        if self.feature_mix_layer:
            delta_flop, x = self.feature_mix_layer.get_flops(x)
            flop += delta_flop
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten

        delta_flop, x = self.classifier.get_flops(x)
        flop += delta_flop
        return flop, x

    def set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                return {
                    'momentum': m.momentum,
                    'eps': m.eps,
                }
        return None

    def init_model(self, model_init, init_div_groups=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weight_parameters(self):
        return self.parameters()

    @staticmethod
    def _make_divisible(v, divisor, min_val=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        :param v:
        :param divisor:
        :param min_val:
        :return:
        """
        if min_val is None:
            min_val = divisor
        new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

def proxyless_base(pretrained=False, net_config=None, net_weight=None, num_classes=1000, dropout=0):
    assert net_config is not None, "Please input a network config"
    # from .proxyless_config import cpu_config,gpu_config,mobile_config,mobile_14_config
    #when run main() please use this one:
    # from proxyless_config import cpu_config,gpu_config,mobile_config,mobile_14_config 
    net_config_json = copy.deepcopy(gpu_config)
    if net_config in 'cpu_config':
        net_config_json = copy.deepcopy(cpu_config)
    if net_config in 'gpu_config':
        net_config_json = copy.deepcopy(gpu_config)
    if net_config in 'mobile_config':
        net_config_json = copy.deepcopy(mobile_config)
    if net_config in 'mobile_14_config':
        net_config_json = copy.deepcopy(mobile_14_config)
#     net_config_json['classifier']['out_features'] = num_classes
    net_config_json['classifier']['dropout_rate'] = dropout
    net = ProxylessNASNets.build_from_config(net_config_json)

    if 'bn' in net_config_json:
        net.set_bn_param(
            bn_momentum=net_config_json['bn']['momentum'],
            bn_eps=net_config_json['bn']['eps'])
    else:
        net.set_bn_param(bn_momentum=0.1, bn_eps=1e-3)

    if pretrained:
        assert net_weight is not None, "Please specify network weights"
        init_path = download_url(net_weight)
        init = torch.load(init_path, map_location='cpu')
        net.load_state_dict(init['state_dict'])

    return net

proxyless_cpu = partial(
    proxyless_base,
    net_config = 'cpu_config',
    net_weight = model_urls['proxyless_cpu'])

mult_prun8_gpu_prun = partial(
    proxyless_base,
    net_config = "gpu_config",
    net_weight = model_urls['proxyless_gpu'])

proxyless_mobile = partial(
    proxyless_base,
    net_config = "mobile_config",
    net_weight = model_urls['proxyless_mobile'])

proxyless_mobile_14 = partial(
    proxyless_base,
    net_config = "mobile_14_config",
    net_weight = model_urls['proxyless_mobile_14'])

if __name__ == '__main__':
    # from compute_flops import print_model_param_nums,count_model_param_flops
    from torchsummary import summary
    model = mult_prun8_gpu_prun(num_classes=1000)
    # print_model_param_nums(model)
    # count_model_param_flops(model)
    print(model)
    summary(model, (3, 112, 112))

    

