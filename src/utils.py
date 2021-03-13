import numpy as np
import json
import pprint
import torch
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import starry


np.random.seed(12)
starry.config.lazy = False
starry.config.quiet = True


def get_light_curve(path_img, ydeg, amp, obl, inc, npts, nrot, sigma=0.005):
    # Initialize map
    map = starry.Map(ydeg=ydeg, reflected=True)
    map.load(path_img)
    map.amp = amp
    map.obl = obl
    map.inc = inc

    # Make the planet rotate 10 times over one full orbit
    time = np.linspace(0, 1, npts)
    theta = np.linspace(0, 360 * nrot, npts)

    # Position of the star relative to the planet in the orbital plane
    t = np.reshape(time, (1, -1))
    p = np.vstack((np.cos(2 * np.pi * t), np.sin(2 * np.pi * t), 0 * t))

    # Rotate to an observer inclination of 60 degrees
    ci = np.cos(map.inc * np.pi / 180)
    si = np.sin(map.inc * np.pi / 180)
    R = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    xs, ys, zs = R.dot(p)

    # Keywords to the `flux` method
    kwargs = dict(theta=theta, xs=xs, ys=ys, zs=zs)

    # Compute the flux
    flux0 = map.flux(**kwargs)
    flux = flux0 + sigma * np.random.randn(npts)

    return flux, map


def get_device(gpu_id=-1):
    if gpu_id >= 0 and torch.cuda.is_available():
        print('device: gpu')
        return torch.device("cuda", gpu_id), True
    else:
        print('device: cpu')
        return torch.device("cpu"), False


class Normalize():
    def __init__(self, fluxes):
        self.mean = np.mean(fluxes, axis=0)
        self.std = np.std(fluxes, axis=0)

    def __call__(self, flux):
        return np.abs(self.mean - flux) / self.std


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
        self.device, use_cuda = get_device(gpu)
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
            flux, _ = get_light_curve(filename, self.cfg['ydeg'], self.cfg['amp'], self.cfg['obl'], 
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
        
    def compare_flux(self, filename, flux_real=None):
        if flux_real is None:
            flux_real, _ = get_light_curve(filename, self.cfg['ydeg'], self.cfg['amp'], self.cfg['obl'], 
                                        self.cfg['inc'], self.cfg['npts'], self.cfg['nrot'])
            
        # generate maps
        self.generate_maps(np.reshape(flux_real, (1, len(flux_real))))
        im = Image.open("tmp_img.png", "r")
        im = im.resize((im.size[0]*2, im.size[1]))
        im.save('tmp_img.png')
        flux_fake, _ = get_light_curve('tmp_img.png', self.cfg['ydeg'], self.cfg['amp'], self.cfg['obl'], 
                                        self.cfg['inc'], self.cfg['npts'], self.cfg['nrot'])
        
        fig, ax = plt.subplots(1, figsize=(12, 4))
        time = np.linspace(0, 1, self.cfg['npts'])
        ax.plot(time, flux_real)
        ax.plot(time, flux_fake)
        ax.set_xlabel("Orbital phase", fontsize=18)
        ax.set_ylabel("Normalized flux", fontsize=18)


