import logging
import torchvision
from dotenv import dotenv_values
import torch
import os
import numpy as np
import random
import argparse
from tqdm import tqdm
from spikingjelly.activation_based import functional, functional
from torch import nn
from source.attacks import fgsm_attack, pgd_attack
from source.model import GeneralUndefendedSNNModel
from source.encoder import Encoder
from source.utils import ToTensorNoDiv

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, help="Set directory of config file. Usually are .env file.")
args = parser.parse_args()

def checkpoint(model, 
               optimizer, 
               ckpt_dir,
               ckpt_name,
               epoch=None, 
               val_loss=None, 
               load_ckpt=True):
    ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}.pth")
    if load_ckpt:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']

    else:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(state_dict, ckpt_path)

def train_step(dataloader, 
               model, 
               loss_fn: nn.Module,
               encoder: Encoder, 
               optimizer):
    
    model.train()
    pbar = tqdm(dataloader)
     
    for img, y in pbar:
        img = encoder(img)
        pred = model(img)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        functional.reset_net(model.snn)
        encoder.reset()
        pbar.set_description(f"Loss: {(loss.item()):>3f}")


def test_step(dataloader, 
              model, 
              loss_fn, 
              epoch, 
              optimizer, 
              config, 
              encoder: Encoder, 
              logger,
              best_loss=None):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()

    for img, y in tqdm(dataloader): 
        with torch.no_grad():

            img = encoder(img)
            pred = model(img)
            functional.reset_net(model.snn)
            encoder.reset()
            
            test_loss += loss_fn(pred, y).item()
            correct += torch.from_numpy(pred.cpu().numpy().argmax(-1) == y.cpu().numpy()).sum().item()
           
    test_loss /= num_batches
    correct /= size
    logger.info(f"Test loss: {(test_loss):>3f}, test acc:{(100*correct):>3f}\n")

    if test_loss < best_loss:
        best_loss = test_loss
        checkpoint(model,
                    optimizer,
                    config["CHECKPOINT_DIR"],
                    "best",
                    epoch=epoch,
                    val_loss=test_loss,
                    load_ckpt=False)
        logger.info(f"Saving best result...")

    checkpoint(model,
                optimizer,
                config["CHECKPOINT_DIR"],
                "latest",
                epoch=epoch,
                val_loss=test_loss,
                load_ckpt=False)
    logger.info(f"Saving latest result...")
    return best_loss

def adv_test_step(dataloader, 
              model, 
              loss_fn,
              encoder: Encoder, 
              logger):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    fgsm_loss, pgd_loss, fgsm_correct, pgd_correct = 0, 0, 0, 0

    for img, y in tqdm(dataloader):
        img = encoder(img)
        encoder.reset()
        model.train()
        fgsm_img = fgsm_attack(loss_fn, model, img, y)
        pgd_img = pgd_attack(loss_fn, model, img, y)
        model.eval()
        
        with torch.no_grad():            
            fgsm_pred = model(fgsm_img)
            functional.reset_net(model.snn)
            
            pgd_pred = model(pgd_img)
            functional.reset_net(model.snn)
            
            fgsm_loss += loss_fn(fgsm_pred, y).item()
            pgd_loss += loss_fn(pgd_pred, y).item()
            fgsm_correct += torch.from_numpy(fgsm_pred.cpu().numpy().argmax(-1) == y.cpu().numpy()).sum().item()
            pgd_correct += torch.from_numpy(pgd_pred.cpu().numpy().argmax(-1) == y.cpu().numpy()).sum().item()
            
    fgsm_loss /= num_batches
    fgsm_correct /= size
    pgd_loss /= num_batches
    pgd_correct /= size
    logger.info(f"FGSM loss: {(fgsm_loss):>3f}, FGSM acc:{(100*fgsm_correct):>3f} PGD loss: {(pgd_loss):>3f}, PGD acc:{(100*pgd_correct):>3f}\n")
	
def preprocess(config):

    TRANSFORM_TRAIN = torchvision.transforms.Compose([
        ToTensorNoDiv(), #Custom division are perform within encoder
        torchvision.transforms.RandomHorizontalFlip(),
    ])

    TRANSFORM_TEST = torchvision.transforms.Compose([
        ToTensorNoDiv(),
    ])

    train_set = torchvision.datasets.CIFAR100(
            root=config["DATADIR"],
            train=True,
            transform=TRANSFORM_TRAIN,
            download=True)

    val_set = torchvision.datasets.CIFAR100(
            root=config["DATADIR"],
            train=False,
            transform=TRANSFORM_TEST,
            download=True)
    
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=int(config["BATCH_SIZE"]), 
                                               shuffle=True,
                                               #generator=torch.Generator(device=config["DEVICE"]),
                                               collate_fn=lambda x: tuple(x_.to(config["DEVICE"]) for x_ in torch.utils.data.dataloader.default_collate(x)))
    test_loader = torch.utils.data.DataLoader(val_set, 
                                              batch_size=int(config["BATCH_SIZE"]),
                                              shuffle=True,
                                              #generator=torch.Generator(device=config["DEVICE"]),
                                              collate_fn=lambda x: tuple(x_.to(config["DEVICE"]) for x_ in torch.utils.data.dataloader.default_collate(x)))
    
    return train_loader, test_loader

def run():

    seed = 3407

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = dotenv_values(args.config)

    train_loader, test_loader = preprocess(config)

    model = GeneralUndefendedSNNModel(int(config["NUM_CLASS"]),
                                      config["VARIANT"],
                                      config["POOL_MODE"],
                                      int(config["LAYER_COUNT"])).to(config["DEVICE"])

    encoder = Encoder(config["ENCODER_TYPE"])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=1e-3, 
                                 weight_decay=0.001)
    
    epoch, test_loss = 1, np.inf

    if not os.path.exists(config["CHECKPOINT_DIR"]): os.makedirs(config["CHECKPOINT_DIR"])

    if os.path.exists(os.path.join(config["CHECKPOINT_DIR"], "best.pth")):
        epoch, test_loss = checkpoint(model,
                                    optimizer,
                                    config["CHECKPOINT_DIR"],
                                    "best")
        print(f"Loaded from {epoch} epoch, avg val loss: {test_loss:>3f}.")

    elif os.path.exists(os.path.join(config["CHECKPOINT_DIR"], "latest.pth")):
        epoch, test_loss = checkpoint(model,
                                    optimizer,
                                    config["CHECKPOINT_DIR"],
                                    "latest")
        print(f"Loaded from {epoch} epoch, avg val loss: {test_loss:>3f}.")

    else: print("Training new model.")

    print(f"Logging into {config["LOG_DIR"]}")
    logging.basicConfig(
        filename=config["LOG_DIR"],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    logger = logging.getLogger()

    for i in range(epoch, int(config["EPOCHS"])+1):
        logger.info(f"Current epoch: {i}.")
        train_step(train_loader, model, loss_fn, encoder, optimizer)
        test_loss = test_step(test_loader, model, loss_fn, (i+1), optimizer, config, encoder, logger, best_loss=test_loss)
        adv_test_step(test_loader, model, loss_fn, encoder, logger)
if __name__ == "__main__":
    run()