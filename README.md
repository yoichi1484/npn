NeuPlaNet dataset / GAN
===

# Dataset
```
data
  ├── n10000_64x32_bin.zip # zip file of map images
  └── flux1 
        ├── flux.npy       # light curve
        └── configs.json   # computation configs of light curves
```

# Setup
```
$ cd data
$ zip n10000_64x32_bin.zip
```

# Usage
Training DCGAN
```
$ python src/dcgan.py \
  --path_lc data/flux1 \
  --path_img data/n10000_64x32_bin \
  --path_log . \
  --gpu_id 0 \
  --batch_size 16 \
  --n_epochs 20000 \
```
