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
Training the map generator
- DCGAN
- CNN + DCGAN
- DeConv
- CNN + DeConv (bset)
```
$ python src/cnn_deconv.py \
  --path_lc data/flux1 \
  --path_img data/n10000_64x32_bin \
  --path_log . \
  --gpu_id 0 \
  --batch_size 16 \
  --n_epochs 20000 \
```

Make flux dataset from images
```
$ python src/make_dataset.py --npts 1000 --path_save data/myflux
```

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
