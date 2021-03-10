import os
import argparse
import json
import numpy as np

from  glob import glob
from tqdm import tqdm
import numpy as np
import starry

np.random.seed(12)
starry.config.lazy = False
starry.config.quiet = True


def make_dataset(path_dataset, n_data, ydeg, amp, obl, inc, npts, nrot):
    path_dataset
    files = sorted(glob("{}/*.png".format(path_dataset)))
    if n_data > 0:
      files = files[:n_data]
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
np.save(args.path_save + "flux", fluxes)
