from  glob import glob
from tqdm import tqdm
import numpy as np
import starry

#from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

np.random.seed(12)
starry.config.lazy = False
starry.config.quiet = True


def make_dataset(path_dataset, n_data, ydeg, amp, obl, inc, npts, nrot):
    path_dataset
    files = sorted(glob("{}/*.png".format(path_dataset)))[:n_data]
    light_curves = []
    for filename in tqdm(files):
        flux, _ = get_light_curve(filename, ydeg, amp, obl, inc, npts, nrot)
        light_curves.append(flux)
    return np.array(light_curves)
  
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

class NeuPlaNet(Dataset):
    def __init__(self, root_dir, fluxes, n_data, img_size, transform=None):
        # 画像ファイルのパス一覧を取得する。
        self.root_dir = root_dir
        self.filenames = sorted(glob("{}/*.png".format(self.root_dir)))[:n_data]
        self.fluxes = fluxes[:n_data]
        self.img_size = img_size
        self.transform = transform

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
        flux = fluxes[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, flux

    def __len__(self):
        return len(self.fluxes)
