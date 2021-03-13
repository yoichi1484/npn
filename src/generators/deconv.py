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
    #adversarial_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss(reduction='sum')
    
    # Show configs
    args.latent_dim = fluxes.shape[1]
    print(json.dumps(args.__dict__, indent=2))
    
    # Make a directory for logging
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.log_dir + "/images", exist_ok=True)
    with open('{}/args.json'.format(args.log_dir), 'w') as f:
        json.dump(args.__dict__, f)
    
    # Initialize generator 
    generator = Generator(args.latent_dim, args.img_size, args.channels)
    
    if use_cuda:
        generator.cuda()
        #adversarial_loss.cuda()
        mse_loss.cuda()
        
    # Initialize weights
    generator.apply(weights_init_normal)
    
    # Optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, 
                                   betas=(args.b1, args.b2))
    
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    # Training
    training_loop = tqdm(range(1, args.n_epochs + 1))
    for epoch in range(args.n_epochs):
        for i, (imgs, flux) in enumerate(dataloader):
            
            if args.no_flux:
                noise = Variable(Tensor(np.random.normal(0, 3, (imgs.shape[0], args.latent_dim))))
            else:
                noise = Variable(flux.type(Tensor))
                
            # Generate fake image batch with G
            fake = generator(noise)
            
            # Compute loss
            imgs = imgs.to(device)
            err = mse_loss(fake, imgs)
            
            # Zero the gradients before running the backward pass
            generator.zero_grad()
            
            # Backward pass: compute gradient of the loss
            err.backward()

            # Update generator
            optimizer_G.step()
            
            n_iter = epoch * len(dataloader) + i
            training_loop.set_description("Epoch %d | Iter %d | Loss: %f" % (epoch, n_iter, err.item()))
        
        # Logging training status
        if epoch % args.log_interval == 0:
            torch.save(generator.state_dict(), "{}/generator.pt".format(args.log_dir))
            save_image(fake.data[:25], "{}/images/{}.png".format(args.log_dir, epoch), nrow=5, normalize=True)


if __name__ == '__main__':
    main()
