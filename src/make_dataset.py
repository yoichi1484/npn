import os
import argparse
import json
import numpy as np

from  glob import glob
from tqdm import tqdm
import numpy as np
import utils


def make_dataset(path_dataset, n_data, ydeg, amp, obl, inc, npts, nrot, fluxes=[], log_interval=100, args=None):
    files = sorted(glob("{}/*.png".format(path_dataset)))
    idx = len(fluxes)
    if n_data > 0:
        files = files[idx:idx+n_data]
        assert len(files) != 0
    else:
        files = files[idx:]
    print("computing flux from [{}] (index: {})".format(files[idx], idx))
    
    assert len(files) != 0
    
    if fluxes!=[]:
        light_curves = flux.tolist()
    else:
        light_curves = []
    
    i = 0
    for filename in tqdm(files): #tqdm(enumerate(files)):
        flux, _ = utils.get_light_curve(filename, ydeg, amp, obl, inc, npts, nrot)
        light_curves.append(flux)
        if args is not None and i % log_interval == 0:
            save(np.array(light_curves), args)
        i += 1
    return np.array(light_curves)


def save(fluxes, args):
    os.makedirs(args.path_save, exist_ok=True)
    with open('{}/configs.json'.format(args.path_save), 'w') as f:
        json.dump(args.__dict__, f)
    np.save(args.path_save + "/flux", fluxes)

    
def check_config(args):
    with open('{}/configs.json'.format(args.path_save)) as f:
        loaded_configs = json.load(f)
    assert loaded_configs["ydeg"] == args.ydeg
    assert loaded_configs["amp"] == args.amp
    assert loaded_configs["obl"] == args.obl
    assert loaded_configs["inc"] == args.inc
    assert loaded_configs["npts"] == args.npts
    assert loaded_configs["nrot"] == args.nrot
    assert loaded_configs["path_img"] == args.path_img
    
    
parser = argparse.ArgumentParser(description='A script of making NeuPlaNet dataset')
parser.add_argument('--path_img', type=str, default="n10000_64x32_bin")
parser.add_argument('--path_save', type=str, default="flux")
parser.add_argument('--ydeg', type=int, default=10)
parser.add_argument('--amp', type=int, default=1.3)
parser.add_argument('--obl', type=int, default=23.5)
parser.add_argument('--inc', type=int, default=60)
parser.add_argument('--n_data', type=int, default=-1)
parser.add_argument('--npts', type=int, default=10000)
parser.add_argument('--nrot', type=int, default=10)
parser.add_argument('--load_flux', action='store_true', default=False,
                    help='flag that does not overwrite flux data')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
args = parser.parse_args()
print(json.dumps(args.__dict__, indent=2))

if args.dry_run:
  n_data = 1
else:
  n_data = args.n_data #-1 # use all images

path = args.path_save + "/flux.npy"
if os.path.exists(path) and args.load_flux:
    check_config(args)
    fluxes = np.load(path)
    print("loaded flux <- {}".format(path))
else:
    fluxes = []

  
print("computing flux")
fluxes = make_dataset(args.path_img, n_data, args.ydeg, args.amp, 
                      args.obl, args.inc, args.npts, args.nrot, fluxes=fluxes, args=args)

print("saving...")
save(fluxes, args)
print("done.")







