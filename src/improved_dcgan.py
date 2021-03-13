import argparse
import os
import math
import json
import datetime
import numpy as np
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from neuplanet import NeuPlaNet
import utils


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        #self.i = 0
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

      
def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_lc", type=str, required=True, help="path of light curve data")
    parser.add_argument("--path_img", type=str, required=True, help="path of map images")
    parser.add_argument("--path_log", type=str, default=".", help="path for logging model")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    #parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--dry_run", action='store_true', help="quickly check a single pass")
    parser.add_argument("--no_flux", action='store_true', help="use the random vector instead of flux")
    parser.add_argument("--gpu_id", type=int, default=-1, help="gpu device id. (cpu = -1)")
    parser.add_argument("--log_interval", type=int, default=100, help="interval between image sampling")
    return parser.parse_args()

def main():
    # Setup arguments
    args = _parse()
    if args.dry_run:
        n_data = 1
        args.n_epochs = 1
    else:
        n_data = -1
    now = str(datetime.datetime.today()).replace(' ', '_').replace(':', '-').replace('.', '_')
    args.log_dir = args.path_log + "/" + now

    #cuda = True if torch.cuda.is_available() else False
    device, use_cuda = utils.get_device(args.gpu_id)

    """
    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    """
    
    # Load flux data
    fluxes = np.load(args.path_lc + "/flux.npy")

    # Setup preprocessing function
    preprocessing = utils.Normalize(fluxes)

    # Configure data loader
    transform = transforms.Compose(
                [transforms.Resize(args.img_size), transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5])]
                )
        
    dataset = NeuPlaNet(
        args.path_img, 
        fluxes = fluxes, 
        n_data = n_data, 
        img_size = args.img_size, 
        transform = transform, 
        preprocessing = preprocessing)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        )
    
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    
    # Show configs
    args.latent_dim = fluxes.shape[1]
    print(json.dumps(args.__dict__, indent=2))
    
    # Initialize generator and discriminator
    generator = Generator(args.latent_dim, args.img_size, args.channels)
    discriminator = Discriminator(args.img_size, args.channels)
    
    if use_cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, 
                                   betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, 
                                   betas=(args.b1, args.b2))
    
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    # Training
    training_loop = tqdm(range(1, args.n_epochs + 1))
    
    # ここから変更
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    #fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)
    
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    
    for epoch in range(args.n_epochs):
        for i, (imgs, flux) in enumerate(dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = imgs.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = adversarial_loss(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
    
            ## Train with all-fake batch
            # Generate batch of latent vectors
            #noise = torch.randn(b_size, args.latent_dim, 1, 1, device=device)
            if args.no_flux:
                noise = Variable(Tensor(np.random.normal(0, 3, (imgs.shape[0], args.latent_dim))))
            else:
                noise = Variable(flux.type(Tensor))
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = adversarial_loss(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizer_D.step()
    
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = adversarial_loss(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizer_G.step()    
            
            n_iter = epoch * len(dataloader) + i
            training_loop.set_description("Epoch %d | Iter %d | G Loss: %f | D Loss: %f" % (epoch, n_iter, errG.item(), errD.item()))
    
        # Make a directory for logging
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.log_dir + "/images", exist_ok=True)
        
        # Logging training status
        if epoch % args.log_interval == 0:
            torch.save(generator.state_dict(), "{}/generator.pt".format(args.log_dir))
            save_image(fake.data[:25], "{}/images/{}.png".format(args.log_dir, epoch), nrow=5, normalize=True)


if __name__ == '__main__':
    main()
