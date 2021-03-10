import numpy as np
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
