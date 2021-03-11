from  glob import glob
import numpy as np

#from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


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
        flux = self.fluxes[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, flux

    def __len__(self):
        return len(self.fluxes)
