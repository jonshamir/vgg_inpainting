import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import torch.optim as optim
from vgg_extractor import get_VGG_features
import argparse

import models_big as models
import datasets

import matplotlib.pyplot as plt
plt.switch_backend('agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--gan_path", type=str, default="./checkpoints/")
    parser.add_argument("--dataset", type=str, default="frogs")
    parser.add_argument("--pretrained_model", type=str, default="frogs_conv")
    parser.add_argument("--test_dir", type=str, default="../test_images/")
    parser.add_argument("--out_dir", type=str, default="../outputs/")
    parser.add_argument("--eval_only", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--deep_context", action='store_true', default=False)

    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--optim_steps", type=int, default=8000)
    parser.add_argument("--prior_weight", type=float, default=10)
    parser.add_argument("--window_size", type=int, default=25)
    parser.add_argument("--feat_layer", type=int, default=5)
    args = parser.parse_args()
    return args

def calc_context_loss(corrupt_images, gen_images, masks):
    # return torch.sum(((corrupt - generated)**2) * masks) # L2
    return torch.sum(torch.abs((corrupt_images - gen_images) * masks)) # L1

def calc_context_loss_deep(corrupt_images, gen_feats, masks, feats_masks, layer=5):
    corrupt_feats = get_VGG_features(corrupt_images, layer=layer).detach()
    # feats_spatial_size = gen_feats.size()[-2:]
    # masks = nn.functional.interpolate(masks, size=feats_spatial_size) * feats_masks
    return torch.sum(torch.abs((corrupt_feats - gen_feats) * masks)) # L1

def inpaint(opt):
    data_path = opt.test_dir + opt.dataset + '/'
    dataset = datasets.CorruptedPatchDataset(data_path, image_size=(opt.image_size, opt.image_size), weighted_mask=True, window_size=opt.window_size)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size)

    # Loading trained GAN model
    saved_G = torch.load(opt.gan_path + opt.pretrained_model + "/modelG.pth")
    saved_D = torch.load(opt.gan_path + opt.pretrained_model + "/modelD.pth")
    saved_E = torch.load(opt.gan_path + opt.pretrained_model + "/modelE.pth")
    saved_Inv = torch.load(opt.gan_path + opt.pretrained_model + "/modelInv.pth")

    netG = models.DeepGenerator(layer=opt.feat_layer).to(device)
    netD = models.DeepDiscriminator(layer=opt.feat_layer).to(device)
    netE = models.DeepEncoder(layer=opt.feat_layer).to(device)
    netInv = models.VGGInverterG(layer=opt.feat_layer).to(device)

    netG.load_state_dict(saved_G, strict=False)
    netD.load_state_dict(saved_D, strict=False)
    netE.load_state_dict(saved_E, strict=False)
    netInv.load_state_dict(saved_Inv, strict=False)

    context_losses = []
    prior_losses = []

    for i, (corrupt_images, original_images, masks, weighted_masks, feats_masks) in enumerate(dataloader):
        corrupt_images, masks, weighted_masks, feats_masks = corrupt_images.to(device), masks.to(device), weighted_masks.to(device), feats_masks.to(device)
        # TEST WITHOUT HOLES
        corrupt_images = original_images.to(device)
        weighted_masks = 1

        # z = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, (corrupt_images.shape[0], opt.latent_dim,))).to(device))
        print("Getting initial noise from encoder...")
        corrupt_feats = get_VGG_features(corrupt_images, layer=opt.feat_layer).detach()
        z = nn.Parameter(netE(corrupt_feats).detach())
        inpaint_opt = optim.Adam([z])

        print("Training input noise...")
        for epoch in range(opt.optim_steps):
            inpaint_opt.zero_grad()
            gen_feats = netG(z)
            gen_images = netInv(gen_feats)

            if opt.deep_context:
                context_loss = calc_context_loss_deep(corrupt_images, gen_feats, weighted_masks, feats_masks, layer=opt.feat_layer)
            else:
                context_loss = calc_context_loss(corrupt_images, gen_images, weighted_masks)

            context_losses.append(context_loss.item())

            prior_loss = context_loss * 0 # torch.mean(netD(gen_feats)) * opt.prior_weight
            prior_losses.append(prior_loss.item())

            inpaint_loss = context_loss + prior_loss

            inpaint_loss.backward()
            inpaint_opt.step()

            if epoch % 100 == 0:
                epoch_str = str(epoch).zfill(6)
                print("Epoch: {}/{}\tLoss: {:.3f}\tContext loss: {:.3f}\tPrior loss: {:.3f}\r".format(1 + epoch, opt.optim_steps, inpaint_loss, context_loss, prior_loss))
                save_image(gen_images, opt.out_dir + "out_{}_{}.png".format(i, epoch_str), normalize=True, range=(-1,1), nrow=5)

        print("")

        blended_images = masks * corrupt_images + (1 - masks) * gen_images.detach()
    
        image_range = torch.min(corrupt_images), torch.max(corrupt_images)
        save_image(corrupt_images, opt.out_dir + "corrupted_{}.png".format(i), normalize=True, range=image_range, nrow=5)
        save_image(gen_images, opt.out_dir + "output_{}.png".format(i), normalize=True, range=image_range, nrow=5)
        save_image(blended_images, opt.out_dir + "blended_{}.png".format(i), normalize=True, range=image_range, nrow=5)
        save_image(original_images, opt.out_dir + "original_{}.png".format(i), normalize=True, range=image_range, nrow=5)

        # save losses plot
        plt.figure(figsize=(10, 5))
        plt.title("Loss During Training")
        plt.plot(context_losses, label="Img Context")
        plt.plot(prior_losses, label="Prior (D)")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(opt.out_dir + "loss.png", dpi=200)

        del z, inpaint_opt

if __name__ == "__main__":
    args = get_arguments()
    inpaint(args)
