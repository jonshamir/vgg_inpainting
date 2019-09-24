import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim

import numpy as np

import argparse

import models
import datasets

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--gan-path", type=str, default="./checkpoints/")
    parser.add_argument("--test-data-dir", type=str, default="../test_images/")
    parser.add_argument("--eval-only", action='store_true', default=False)
    parser.add_argument("--test-only", action='store_true', default=False)

    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--optim-steps", type=int, default=1500)
    parser.add_argument("--blending-steps", type=int, default=3000)
    parser.add_argument("--prior-weight", type=float, default=0.003)
    parser.add_argument("--window-size", type=int, default=25)
    args = parser.parse_args()
    return args

def calc_context_loss(corrupted_images, generated_images, masks, weighted=True):
    return torch.sum(((corrupted_images-generated_images)**2)*masks)

def image_gradient(image):
    a = torch.Tensor([[[[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]],
                       [[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]],
                       [[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]]
                       ]]).cuda()
    G_x = F.conv2d(image, a, padding=1)
    b = torch.Tensor([[[[ 1, 2, 1],
                        [ 0, 0, 0],
                        [-1,-2,-1]],
                       [[1, 2, 1],
                        [0, 0, 0],
                        [-1,-2,-1]],
                       [[1, 2, 1],
                        [0, 0, 0],
                        [-1,-2,-1]]
                       ]]).cuda()
    G_y = F.conv2d(image, b, padding=1)
    return G_x, G_y

def inpaint(args):
    dataset = datasets.RandomPatchDataset(args.test_data_dir, image_size=(args.image_size, args.image_size), weighted_mask=True, window_size=args.window_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Loading trained GAN model
    saved_G = torch.load(args.gan_path + "modelG.pth")
    saved_D = torch.load(args.gan_path + "modelD.pth")
    netG = models.BasicGenerator().cuda()
    netD = models.BasicDiscriminator().cuda()
    netG.load_state_dict(saved_G)
    netD.load_state_dict(saved_D)

    for i, (corrupted_images, original_images, masks, weighted_masks) in enumerate(dataloader):
        corrupted_images, masks, weighted_masks = corrupted_images.cuda(), masks.cuda(), weighted_masks.cuda()
        z_optimum = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, (corrupted_images.shape[0],args.latent_dim,))).cuda())
        optimizer_inpaint = optim.Adam([z_optimum])

        print("Training input noise...")
        for epoch in range(args.optim_steps):
            optimizer_inpaint.zero_grad()
            generated_images = netG(z_optimum)
            D_loss = netD(generated_images)
            context_loss = calc_context_loss(corrupted_images, generated_images, weighted_masks)
            prior_loss = torch.sum(-torch.log(D_loss))
            inpaint_loss = context_loss + args.prior_weight * prior_loss
            inpaint_loss.backward()
            optimizer_inpaint.step()
            if epoch % 500 == 0:
                print("[Epoch: {}/{}] \t[Loss: \t[Context: {:.3f}] \t[Prior: {:.3f}] \t[Inpaint: {:.3f}]]  \r".format(1+epoch, args.optim_steps, context_loss, prior_loss, inpaint_loss))
        print("")

        blended_images = masks * corrupted_images + (1 - masks) * generated_images.detach()
    
        image_range = torch.min(corrupted_images), torch.max(corrupted_images)
        save_image(corrupted_images, "../outputs/corrupted_{}.png".format(i), normalize=True, range=image_range, nrow=5)
        save_image(generated_images, "../outputs/output_{}.png".format(i), normalize=True, range=image_range, nrow=5)
        save_image(blended_images, "../outputs/blended_{}.png".format(i), normalize=True, range=image_range, nrow=5)
        save_image(original_images, "../outputs/original_{}.png".format(i), normalize=True, range=image_range, nrow=5)

        del z_optimum, optimizer_inpaint

if __name__ == "__main__":
    args = get_arguments()
    inpaint(args)
