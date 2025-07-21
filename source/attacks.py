import torch
from spikingjelly.activation_based import functional, functional

def fgsm_attack(loss_fn, 
                model, 
                images, 
                labels,
                epsilon=6/255, 
                device="cuda" if torch.cuda.is_available() else "cpu"):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
    #hooks = register_hooks(model)

    outputs = model(images)
    loss = loss_fn(outputs, labels)
    
    loss.backward(retain_graph=True)
    grad = images.grad.data

    grad_sign = grad.sign()

    adv_images = images + epsilon * grad_sign
    adv_images = torch.clamp(adv_images, 0, 1)  # keep pixel range
    model.zero_grad()

    functional.reset_net(model.snn)

    #for h in hooks: h.remove()
    return adv_images

def pgd_attack(loss_fn, 
               model, 
               images, 
               labels,
               epsilon=6/255, 
               alpha=0.01, 
               iters=7, 
               device="cuda" if torch.cuda.is_available() else "cpu"):
    
    ori_images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images = ori_images + torch.empty_like(ori_images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, 0, 1)

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)

        loss = loss_fn(outputs, labels)
        loss.backward(retain_graph=True)

        grad = images.grad.data
        if torch.isnan(grad).any().item() or torch.isinf(grad).any().item() or grad.norm().item() < 1e-6: 
            raise Exception("Abnormal grad norm detected")
        grad_sign = grad.sign()

        adv_images = images + alpha * grad_sign
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, 0, 1).detach_()
        model.zero_grad()

        functional.reset_net(model.snn)
         
    return images