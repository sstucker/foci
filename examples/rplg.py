# -*- coding: utf-8 -*-
"""
Simulate radially-polarized Laguerre-Gaussian beams

References
[1] Kozawa, Y., Matsunaga, D. & Sato, S. Superresolution imaging via superoscillation focusing of a radially polarized beam. Optica. (2018).

"""

import numpy as np
from numpy import pi as PI
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys
import time

from scipy.special import genlaguerre as L

try:
    from foci import debye_wolf
    from foci import UM, NM, MM
except ModuleNotFoundError:  # If not installed, assume user is running from \examples
    sys.path.insert(0, os.path.abspath('..'))
    sys.path.insert(0, os.path.join(os.path.abspath('..'), 'foci'))
    from foci import debye_wolf
    from foci import UM, NM, MM


def lgbeam(r, p, beta, pupil_diameter):
    """Laguerre-Gaussian beam as in [1]"""
    D4sigma = pupil_diameter / beta
    w0 = D4sigma / (2 * np.sqrt(2*p + 2))
    return r/w0 * np.exp(-r**2/w0**2) * L(p, 1)((2*r**2) / w0**2)


# %% Generate objective forward model

# Lens parameters
PUPIL_DIAMETER = 15 * MM
FOCAL_LENGTH = 2 * MM  # 100X
WAVELENGTH = 532 * NM

# Field parameters
FIELD_WIDTH = 6 * UM
FIELD_DEPTH = 6 * UM
FIELD_INDEX = 1.45  # Silicon Oil

# Simulation resolution
N = 256
Z = 256

print('Precomputing forward model of Olympus UPLSAPO 100XO...')
obj = debye_wolf.Objective(WAVELENGTH, FIELD_INDEX, FOCAL_LENGTH, PUPIL_DIAMETER, N, FIELD_WIDTH)
print('...simulated objective with NA={:.3f}'.format(obj._na))

# %% Calculate 3rd order radially-polarized Laguerre-Gaussian pupil

rpos = np.linspace(-PUPIL_DIAMETER / 2, PUPIL_DIAMETER / 2, N)
xx, yy = np.meshgrid(rpos, rpos)
r = np.sqrt(xx**2 + yy**2)

BETA = 0.845
pupil_x = lgbeam(r, 3, BETA, PUPIL_DIAMETER)
pupil = debye_wolf.VectorialPupil(pupil_x=pupil_x, diameter=PUPIL_DIAMETER)
pupil = debye_wolf.vortical_polarize(pupil, angle=0)

start = time.time()
field = obj.focus(pupil, Z, FIELD_DEPTH)
elapsed = time.time() - start
print('Simulated {} planes, {:.3f} ms per plane, {:.3f} s total.'.format(Z, (elapsed * 1000) / Z, elapsed))
psf = field.intensity()

# %% Generate figures

pupil.display(downsample=7, cmap_amp='hot', display_phase=False)
plt.savefig('rplg-example-pupil')

plt.figure('RP-LG PSF')
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(
    psf[:, :, Z // 2],
    cmap='hot',
    extent=(-FIELD_WIDTH / 2, FIELD_WIDTH / 2, -FIELD_WIDTH / 2, FIELD_WIDTH / 2)
)
ax1.add_patch(Rectangle((1.5 * UM, -2.6 * UM), UM, 0.2 * UM, color='white'))  # 1 micron scale bar
ax1.axis('off')
ax2 = plt.subplot(1, 2, 2)
ax2.imshow(
    np.rot90(psf[N // 2, :, :]),
    cmap='hot',
    extent=(-FIELD_WIDTH / 2, FIELD_WIDTH / 2, -FIELD_DEPTH / 2, FIELD_DEPTH / 2)
)
ax2.axis('off')
plt.tight_layout()
plt.savefig('rplg-example-psf')
