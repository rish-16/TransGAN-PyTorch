# TransGAN-PyTorch
PyTorch implementation of the TransGAN paper

The original paper can be found [here](https://arxiv.org/abs/2102.07074).

### Installation
You can install the package via `pip`:

```bash
pip install transgan-pytorch
```

### Usage

```python
import torch
from transgan_pytorch import TransGAN

tgan = TransGAN(...)

z = torch.rand(100) # random noise
pred = tgan(z)
```

### License
[MIT](https://github.com/rish-16/TransGAN-PyTorch/blob/main/LICENSE)