import os
import argparse
import json
import numpy as np

from  glob import glob
from tqdm import tqdm
import numpy as np
import utils


def make_dataset(path_dataset, n_data, ydeg, amp, obl, inc, npts, nrot):
    path_dataset
    files = sorted(glob("{}/*.png".format(path_dataset)))
    if n_data > 0:
      files = files[:n_data]
    assert len(files) != 0
    light_curves = []
    for filename in tqdm(files):
        flux, _ = utils.get_light_curve(filename, ydeg, amp, obl, inc, npts, nrot)
        light_curves.append(flux)
    return np.array(light_curves)

parser = argparse.ArgumentParser(description='A script of making NeuPlaNet dataset')
parser.add_argument('--path_img', type=str, default="n10000_64x32_bin")
parser.add_argument('--path_save', type=str, default="flux")
parser.add_argument('--ydeg', type=int, default=10)
parser.add_argument('--amp', type=int, default=1.3)
parser.add_argument('--obl', type=int, default=23.5)
parser.add_argument('--inc', type=int, default=60)
parser.add_argument('--npts', type=int, default=10000)
parser.add_argument('--nrot', type=int, default=10)
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
args = parser.parse_args()
print(json.dumps(args.__dict__, indent=2))

if args.dry_run:
  n_data = 1
else:
  n_data = -1 # use all images
  
print("computing flux")
fluxes = make_dataset(args.path_img, n_data, args.ydeg, args.amp, 
                      args.obl, args.inc, args.npts, args.nrot)

print("saving...")
os.makedirs(args.path_save, exist_ok=True)
with open('{}/configs.json'.format(args.path_save), 'w') as f:
  json.dump(args.__dict__, f)
np.save(args.path_save + "/flux", fluxes)
print("done.")
