import torch
import numpy as np
from torch import nn

from torchvision.transforms import functional as F
from torchvision.transforms.functional import F_pil
import sys

try:
    import accimage
except ImportError:
    accimage = None

def to_tensor_no_div(pic):
    default_float_dtype = torch.get_default_dtype()

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic).to(dtype=default_float_dtype)

    # handle PIL Image
    mode_to_nptype = {"I": np.int32, "I;16" if sys.byteorder == "little" else "I;16B": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))

    if pic.mode == "1":
        img = 255 * img
    img = img.view(pic.size[1], pic.size[0], F_pil.get_image_num_channels(pic))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=default_float_dtype)
    else:
        return img
    
class ToTensorNoDiv(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, image):
        return to_tensor_no_div(image)

class PILToTensor(nn.Module):
    def __init__(self, div=None):
        super().__init__()
        self.div = div
    
    def forward(self, image):
        image = image.convert(mode='RGB')
        image = F.pil_to_tensor(image)
        if self.div: image = image.float()/self.div
        return image
    
def register_hooks(model):
    hooks = []

    def hook_fn(module, grad_input, grad_output):
        print(f"--- {module.__class__.__name__} ---")
        if grad_input:
            nonzero = (grad_input[0] != 0).sum().item()
            total = grad_input[0].numel()
            print(f"Input grad: {total - nonzero}/{total} zeros ({100 * (total - nonzero)/total:.2f}%)")
        if grad_output:
            nonzero = (grad_output[0] != 0).sum().item()
            total = grad_output[0].numel()
            print(f"Output grad: {total - nonzero}/{total} zeros ({100 * (total - nonzero)/total:.2f}%)")
        print(f"Grad norm: {grad_input[0].norm():.2f}")

    for module in model.modules():
        if len(list(module.children())) == 0:  # leaf modules only
            hooks.append(module.register_full_backward_hook(hook_fn))

    return hooks  # store this to remove hooks later