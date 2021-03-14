from  glob import glob
import numpy as np
import json
import pprint

#from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

import utils


class NeuPlaNet(Dataset):
    def __init__(self, root_dir, fluxes, n_data, img_size, noise=0.0, transform=None, preprocessing=None):
        # 画像ファイルのパス一覧を取得する。
        self.root_dir = root_dir
        self.filenames = sorted(glob("{}/*.png".format(self.root_dir)))[:n_data]
        self.fluxes = fluxes[:n_data]
        self.img_size = img_size
        self.noise = noise
        self.transform = transform
        self.preprocessing = preprocessing

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, flux)
        """
        # pngなので、jpgのRGBにしてからLで二値化
        img = Image.open(self.filenames[index]).convert('RGB').convert('L')  #os.path.join(self.root_dir, self.filenames[index]))
        img = img.resize((self.img_size, self.img_size))
        flux = self.fluxes[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.preprocessing is not None:
            flux = self.preprocessing(flux)
            
        flux = self.add_noise(flux)

        return img, flux

    def __len__(self):
        return len(self.fluxes)
    
    def add_noise(self, flux):
        return flux + self.noise * np.random.randn(*flux.shape)

    
class NeuPlaNetGenerator():
    def __init__(self, path_model, generator, preprocessing, path_lc=None, gpu=-1):
        with open("{}/args.json".format(path_model)) as f:
            self.args = json.load(f)
        
        if path_lc is None:
            path_lc = self.args["path_lc"]
        with open("{}/configs.json".format(path_lc)) as f:
            self.cfg = json.load(f)
            
        print("\nconfigs of trained model")
        pprint.pprint(self.args, width=40)
        print("\nconfigs of computing the flux")
        pprint.pprint(self.cfg, width=40)
                
        self.generator = generator
        self.device, use_cuda = utils.get_device(gpu)
        self.generator.load_state_dict(torch.load(path_model + "/generator.pt", map_location=self.device))
        #self.Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        #with open("drive/MyDrive/neuplanet/data/flux1/configs.json") as f:
        self.preprocessing = preprocessing


    def generate_maps(self, fluxes):
        fluxes = np.array([self.preprocessing(flux) for flux in fluxes])
        z = torch.from_numpy(fluxes).float().to(self.device)
        with torch.no_grad():
            maps = self.generator(z)
        save_image(maps.data, "tmp_img.png", nrow=5, normalize=True)
        return maps

    def compare_maps(self, filename, flux=None):
        # compute flux
        
        #if fluxes is None:
        #    fluxes = []
        #    for path_imgm in zip(filenames):
        #        flux = utils.get_light_curve(path_img, self.cfg['ydeg'], self.cfg['amp'], self.cfg['obl'], 
        #                                self.cfg['inc'], self.cfg['npts'], self.cfg['nrot'])
        #        fluxes.append(flux)
        #    fluxes = np.array(fluxes)
        if flux is None:
            flux, _ = utils.get_light_curve(filename, self.cfg['ydeg'], self.cfg['amp'], self.cfg['obl'], 
                                        self.cfg['inc'], self.cfg['npts'], self.cfg['nrot'])
        flux = np.reshape(flux, (1, len(flux)))

        # generate maps
        print("generated")
        self.generate_maps(flux)
        im = Image.open("tmp_img.png", "r")
        im = im.resize((im.size[0]*2, im.size[1]))
        plt.imshow(np.array(im))
        plt.show()

        # real maps
        print("real")
        im = Image.open(filename)#.convert('RGB').convert('L') 
        plt.imshow(np.array(im))
        plt.show()
        
    def compare_light_curves(self, filename, flux_real=None):
        if flux_real is None:
            flux_real, _ = utils.get_light_curve(filename, self.cfg['ydeg'], self.cfg['amp'], self.cfg['obl'], 
                                        self.cfg['inc'], self.cfg['npts'], self.cfg['nrot'])
            
        # generate maps
        self.generate_maps(np.reshape(flux_real, (1, len(flux_real))))
        im = Image.open("tmp_img.png", "r")
        im = im.resize((im.size[0]*2, im.size[1]))
        im.save('tmp_img.png')
        flux_fake, _ = utils.get_light_curve('tmp_img.png', self.cfg['ydeg'], self.cfg['amp'], self.cfg['obl'], 
                                        self.cfg['inc'], self.cfg['npts'], self.cfg['nrot'])
        
        fig, ax = plt.subplots(1, figsize=(12, 4))
        time = np.linspace(0, 1, self.cfg['npts'])
        ax.plot(time, flux_real)
        ax.plot(time, flux_fake)
        ax.set_xlabel("Orbital phase", fontsize=18)
        ax.set_ylabel("Normalized flux", fontsize=18)
