from typing import Any
import flax.linen as nn


Array = Any


class NetLinLayer(nn.Module):
    features: int = 1
    use_dropout: bool = False
    training: bool = False

    @nn.compact
    def __call__(self, x):
        if self.use_dropout:
            x = nn.Dropout(rate=0.5)(x, deterministic=not self.training)
        x = nn.Conv(self.features, (1, 1), padding=0, use_bias=False)(x)
        return x 


class AlexNet(nn.Module):

    @nn.compact
    def __call__(self, x: Array):
        x = nn.Conv(64, (11, 11), strides=(4, 4), padding=(2, 2))(x)
        x = nn.relu(x)
        relu_1 = x
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = nn.Conv(192, (5, 5), padding=2)(x)
        x = nn.relu(x)
        relu_2 = x
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = nn.Conv(384, (3, 3), padding=1)(x)
        x = nn.relu(x)
        relu_3 = x
        x = nn.Conv(256, (3, 3), padding=1)(x)
        x = nn.relu(x)
        relu_4 = x
        x = nn.Conv(256, (3, 3), padding=1)(x)
        x = nn.relu(x)
        relu_5 = x

        return [relu_1, relu_2, relu_3, relu_4, relu_5]

        
class VGG16(nn.Module):
    
    @nn.compact
    def __call__(self, x: Array):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 
               'M', 512, 512, 512, 'M', 512, 512, 512]

        layer_ids = [1, 4, 8, 12, 16]
        out = []
        for i, v in enumerate(cfg):
            if v == 'M':
                x = nn.max_pool(x, (2, 2,), strides=(2, 2))
            else:
                x = nn.Conv(v, (3, 3), padding=(1, 1))(x)
                x = nn.relu(x)
                if i in layer_ids:
                    out.append(x)
        return out
