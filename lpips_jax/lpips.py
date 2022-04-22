import os
import inspect
from typing import Optional, Any
import pickle
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from .models import *


Dtype = Any


class LPIPSEvaluator:
    def __init__(self, replicate=True, pretrained=True, net='alexnet', lpips=True,
                 use_dropout=True, dtype=jnp.float32):
        self.lpips = LPIPS(pretrained, net, lpips, use_dropout,
                           training=False, dtype=dtype)
        model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', f'weights/{net}.ckpt'))
        self.params = pickle.load(open(model_path, 'rb'))
        if replicate:
            self.params = flax.jax_utils.replicate(self.params)
        self.params = dict(params=self.params)
        
        self.replicate = replicate
    
    def __call__(self, images_0, images_1):
        fn = jax.pmap(self.lpips.apply) if self.replicate else self.lpips.apply
        return fn(
            self.params,
            images_0,
            images_1
        ) 


class LPIPS(nn.Module):
    pretrained: bool = True
    net_type: str = 'alexnet'
    lpips: bool = True
    use_dropout: bool = True
    training: bool = False
    dtype: Optional[Dtype] = jnp.float32

    @nn.compact
    def __call__(self, images_0, images_1):
        shift = jnp.array([-0.030, -0.088, -0.188], dtype=self.dtype)
        scale = jnp.array([0.458, 0.448, 0.450], dtype=self.dtype)
        images_0 = (images_0 - shift) / scale
        images_1 = (images_1 - shift) / scale
        
        if self.net_type == 'alexnet':
            net = AlexNet()
        elif self.net_type == 'vgg16':
            net = VGG16()
        else:
            raise ValueError(f'Unsupported net_type: {self.net_type}. Must be in [alexnet, vgg16]')
        
        outs_0, outs_1 = net(images_0), net(images_1)
        diffs = []
        for feat_0, feat_1 in zip(outs_0, outs_1):
            diff = (normalize(feat_0) - normalize(feat_1)) ** 2
            diffs.append(diff)
        
        res = []
        for d in diffs:
            if self.lpips:
                d = NetLinLayer(use_dropout=self.use_dropout)(d)
            else:
                d = jnp.sum(d, axis=-1, keepdims=True)
            d = spatial_average(d, keepdims=True)
            res.append(d)

        val = sum(res)
        return val


def spatial_average(feat, keepdims=True):
    return jnp.mean(feat, axis=[1, 2], keepdims=keepdims)


def normalize(feat, eps=1e-10):
    norm_factor = jnp.sqrt(jnp.sum(feat ** 2, axis=-1, keepdims=True))
    return feat / (norm_factor + eps)
