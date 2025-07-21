# grad-obf-snn
Gradient Partial Obfuscation in Spiking Neural Network Adversarial Benchmarking

## Quickstart

### Step 1: Environment setup and repo download

To setup the environment testing with this encoder, you will need Pytorch and SpikingJelly. We suggest using conda environment with:

```bash
$ conda create -n env python=3.12.2
$ conda install pytorch=2.3.0 torchvision=0.18.0 torchaudio=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia #As latest pytorch conda guide, change cuda version suitable to your case.
$ pip install spikingjelly
```

or clone and modify locally:

```bash
$ git clone https://github.com/luutn2002/grad-obf-snn.git
```

### Step 2: Usage

Test train model using configs:
```bash
$ python3 train.py -c ./configs/cifar100/max/.env.sewrn50.phase
```

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