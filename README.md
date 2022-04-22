# LPIPS-Jax
Jax port of the original [PyTorch](https://github.com/richzhang/PerceptualSimilarity) implementation of [LPIPS](https://richzhang.github.io/PerceptualSimilarity/). The current version supports pretrained AlexNet and VGG16, and pretrained linear layers.

# Installation
`pip install lpips-jax`

# Usage
For `replicate=False`:
```python
import lpips_jax
import numpy as np

images_0 = np.random.randn(4, 224, 224, 3)
images_1 = np.random.randn(4, 224, 224, 3)

lpips = lpips_jax.LPIPSEvaluator(replicate=False, net='alexnet') # ['alexnet', 'vgg16']
out = lpips(images_0, images_1)
```

For `replicate=True`
```python
import lpips_jax
import numpy as np
import jax

n_devices = jax.local_device_count()
images_0 = np.random.randn(n_devices, 4, 224, 224, 3)
images_1 = np.random.randn(n_devices, 4, 224, 224, 3)

# replicate=True is the default setting
lpips = lpips_jax.LPIPSEvaluator(net='alexnet') # ['alexnet', 'vgg16]
out = lpips(images_0, images_1) # internally calls jax.pmap
```
