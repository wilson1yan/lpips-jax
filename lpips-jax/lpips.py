from typing import Optional, Any
import jax.numpy as jnp
import flax.linen as nn
from .models import *


Dtype = Any


class LPIPS(nn.Module):
    pretrained: bool = True
    net_type: str = 'alex'
    lpips: bool = True,
    use_dropout: bool = True
    training: bool = False
    dtype: Optional[Dtype] = jnp.float32

    @nn.compact
    def __call__(self, images_0, images_1):
        shift = jnp.array([-0.030, -0.088, -0.188], dtype=self.dtype)
        scale = jnp.array([0.458, 0.448, 0.450], dtype=self.dtype)
        images_0 = (images_0 - shift) / scale
        images_1 = (images_1 - shift) / scale
        
        if self.net_type == 'alex':
            net = AlexNet()
        elif self.net_type in ['vgg', 'vgg16']:
            net = VGG16()
        else:
            raise ValueError(f'Unsupported net_type: {self.net_type}. Must be in [alexnet, vgg, vgg16]')
        
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
                d = jnp.sum(d, axis=-1, keepdim=True)
            d = spatial_average(d, keepdim=True)
            res.append(d)

        val = sum(res)
        return val
        

def spatial_average(feat, keepdim=True):
    return jnp.mean(feat, axis=[1, 2], keepdim=keepdim)


def normalize(feat, eps=1e-10):
    norm_factor = jnp.sqrt(jnp.sum(feat ** 2, axis=-1, keepdim=True))
    return feat / (norm_factor + eps)