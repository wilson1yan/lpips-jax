import lpips
import lpips_jax
import torch
import numpy as np


images_0 = np.random.randn(4, 224, 224, 3)
images_1 = np.random.randn(4, 224, 224, 3)

# PyTorch
device = torch.device('cuda')
lp_torch = lpips.LPIPS(net='vgg').to(device)
torch.set_grad_enabled(False)
out_torch = lp_torch(torch.FloatTensor(images_0).movedim(-1, 1).to(device),
                     torch.FloatTensor(images_1).movedim(-1, 1).to(device))
out_torch = out_torch.cpu().numpy()


# Flax / Jax
lp_jax = lpips_jax.LPIPSEvaluator(replicate=False, net='vgg16')
out_jax = lp_jax(images_0, images_1)

assert np.allclose(out_torch, out_jax), np.max(np.abs(out_torch - out_jax))
print('Passed')
