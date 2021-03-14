NeuPlaNet dataset / GAN
===

# Dataset
```
data
  ├── n10000_64x32_bin.zip # zip file of map images
  └── flux1 
        ├── flux.npy       # light curve
        └── configs.json   # configs of computating light curves
```

# Setup
```
$ cd data
$ zip n10000_64x32_bin.zip
```

# Usage
## Training 
Training the map generator
- DCGAN
- CNN + DCGAN
- DeConv
- CNN + DeConv (bset)
```
$ python src/cnn_deconv.py \
  --path_lc data/flux \
  --path_img data/n10000_64x32_bin \
  --path_log log \
  --gpu_id 0 \
  --batch_size 512 \
  --n_epochs 500 \
  # --dry_run # For test run
```

## Predict the map from flux
Setup ```NeuPlaNetGenerator```
```python
import numpy as np
from glob import glob
import torchvision
import src.cnn_deconv as model
import src.utils as utils

model_path = "log/[MODEL DIR]"
lc_path = "data/flux"
path_img = "n10000_64x32_bin"
fluxes = np.load(lc_path + "/flux.npy")
filenames = sorted(glob("{}/*.png".format(path_img)))

generator = model.Generator(npts=1000, latent_dim=1000, img_size=64, 
                            channels=1)
preprocessing = utils.Normalize(fluxes)
npn = utils.NeuPlaNetGenerator(model_path, generator, preprocessing)
```
Testing
```python
idx = 0
filename = filenames[idx]
flux = fluxes[idx]
npn.compare_maps(filename, flux)
npn.compare_flux(filename, flux)
```

## Utilities
Compute the flux from an image
```python
from src.utils import get_light_curve
flux, map = get_light_curve("n10000_64x32_bin/map_00294.png", 
                             ydeg = 10, 
                             amp = 1.3, 
                             obl = 23.5, 
                             inc = 60, 
                             npts = 1000, 
                             nrot = 10)
```

Make flux dataset from images
```
$ python src/make_dataset.py --npts 1000 --path_save data/myflux
```

Plot a light curve
```python
from numpy as np
import matplotlib.pyplot as plt
import starry

np.random.seed(12)
starry.config.lazy = False
starry.config.quiet = True

npts = 1000
time = np.linspace(0, 1, npts)
fig, ax = plt.subplots(1, figsize=(12, 4))
ax.plot(time, flux)
ax.set_xlabel("orbital phase", fontsize=18)
ax.set_ylabel("flux", fontsize=18);
```
