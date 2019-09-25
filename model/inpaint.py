import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim

import numpy as np

import argparse

import models_big as models
import datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--gan_path", type=str, default="./checkpoints/")
    parser.add_argument("--d_name", type=str, default="modelD")
    parser.add_argument("--g_name", type=str, default="modelG")
    parser.add_argument("--test_data_dir", type=str, default="../test_images/")
    parser.add_argument("--eval_only", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)

    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--optim_steps", type=int, default=8000)
    parser.add_argument("--prior_weight", type=float, default=1)
    parser.add_argument("--window_size", type=int, default=25)
    args = parser.parse_args()
    return args

def calc_context_loss(corrupt, generated, masks):
    # return torch.sum(((corrupt - generated)**2) * masks)
    return torch.sum(torch.abs((corrupt - generated) * masks))

def inpaint(args):
    dataset = datasets.CorruptedPatchDataset(args.test_data_dir, image_size=(args.image_size, args.image_size), weighted_mask=True, window_size=args.window_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Loading trained GAN model
    saved_G = torch.load(args.gan_path + args.g_name + ".pth")
    saved_D = torch.load(args.gan_path + args.d_name + ".pth")
    netG = models.BasicGenerator().to(device)
    netD = models.BasicDiscriminator().to(device)
    netG.load_state_dict(saved_G)
    netD.load_state_dict(saved_D)

    for i, (corrupted_images, original_images, masks, weighted_masks) in enumerate(dataloader):
        corrupted_images, masks, weighted_masks = corrupted_images.to(device), masks.to(device), weighted_masks.to(device)
        z = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, (corrupted_images.shape[0],args.latent_dim,))).to(device))
        inpaint_opt = optim.Adam([z])

        print("Training input noise...")
        for epoch in range(args.optim_steps):
            inpaint_opt.zero_grad()
            generated_images = netG(z)
            context_loss = calc_context_loss(corrupted_images, generated_images, weighted_masks)
            prior_loss = torch.mean(netD(generated_images)) * args.prior_weight
            inpaint_loss = context_loss + prior_loss
            inpaint_loss.backward()
            inpaint_opt.step()
            if epoch % 500 == 0:
                print("[Epoch: {}/{}]\tContext loss: {:.3f}\tPrior loss: {:.3f}\tInpaint: {:.3f}\r".format(1+epoch, args.optim_steps, context_loss, prior_loss, inpaint_loss))
        print("")

        blended_images = masks * corrupted_images + (1 - masks) * generated_images.detach()
    
        image_range = torch.min(corrupted_images), torch.max(corrupted_images)
        save_image(corrupted_images, "../outputs/corrupted_{}.png".format(i), normalize=True, range=image_range, nrow=5)
        save_image(generated_images, "../outputs/output_{}.png".format(i), normalize=True, range=image_range, nrow=5)
        save_image(blended_images, "../outputs/blended_{}.png".format(i), normalize=True, range=image_range, nrow=5)
        save_image(original_images, "../outputs/original_{}.png".format(i), normalize=True, range=image_range, nrow=5)

        del z, inpaint_opt

if __name__ == "__main__":
    args = get_arguments()
    inpaint(args)
