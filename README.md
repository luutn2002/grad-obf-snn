# grad-obf-snn
Input gradient dense-sparse trade off is the adversarial robustness-generalization trade off in spiking neural network optimization

## Overview

We explore the natural robustness of SNN through experimenting and defer a correlation between adversarial robustness-generalization trade off of gradient sparsity.

## Quickstart

### Step 1: Environment setup and requirements installation

To setup the environment, you will need [Pytorch](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) and [SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/#index-en). We suggest using conda environment with:

```bash
$ conda create -n env python=3.12.2
$ conda install pytorch=2.3.0 torchvision=0.18.0 torchaudio=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia #As latest pytorch conda guide, change cuda version suitable to your case.
$ pip install spikingjelly
```

then clone the repo:

```bash
$ git clone https://github.com/luutn2002/grad-obf-snn.git
```

### Step 2: Usage

To ensure reproducibility, remember to use static random seed:
```python
import torch
import numpy as np
import random

seed = 3407 # Based on https://arxiv.org/abs/2109.08203

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Test train models from our experiments, you can use available configs included:
```bash
$ python3 train.py -c ./configs/cifar100/max/.env.sewrn50.phase
```
## Input-output gradient tracking for analysis

To track input gradient in attacks, you can modify attacks as:
```python
    from source.utils import register_hooks #Import hook register
    ... #Inside attack function
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
    hooks = register_hooks(model) #Register gradient hook

    outputs = model(images)
    loss = loss_fn(outputs, labels)

    loss.backward(retain_graph=True)
    grad = images.grad.data

    grad_sign = grad.sign()

    adv_images = images + epsilon * grad_sign
    adv_images = torch.clamp(adv_images, 0, 1)  # keep pixel range
    model.zero_grad()

    functional.reset_net(model.snn)

    for h in hooks: h.remove() #Remove hook
    return adv_images
```
## Used datasets

All used dataset is included in [torchvision](https://docs.pytorch.org/vision/main/datasets.html).

## License

Source code is licensed under MIT License.

## Contribution guidelines

Please open an issue or pull request if there are bugs or contribution to be made. Thank you.

## Others
Pytorch [guides](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) and SpikingJelly [guides](https://spikingjelly.readthedocs.io/zh-cn/latest/#index-en) are available.

## Citations
Paper is under review. Temporarily please cite as:
```bibtex
@misc{luu2025parameter,
  author = {Luu T. Nhan},
  title = {Input gradient dense-sparse trade off is the adversarial robustness-generalization trade off in spiking neural network optimization},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/luutn2002/grad-obf-snn}},
}
```
