# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:12:12 2023

@author: tuckes06
"""

import numpy as np
from numpy import pi as PI
import matplotlib.pyplot as plt
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


def gauss2d(n, waist):
    """Return a 2D gaussian profile with 1/e distance `waist`"""
    r = np.linspace(-n / 2, n / 2, n)
    xx, yy = np.meshgrid(r, r)
    return np.exp((-2 * np.sqrt(xx**2 + yy**2)**2) / waist**2)
    

# %% Generate objective forward model

# Lens parameters
PUPIL_DIAMETER = 15 * MM
FOCAL_LENGTH = 14.85 * MM
WAVELENGTH = 920 * NM

# Field parameters
FIELD_WIDTH = 10 * UM
FIELD_DEPTH = 10 * UM
FIELD_INDEX = 1.33  # Silicon Oil

# Simulation resolution
N = 256
Z = 256

print('Precomputing forward model of Olympus 10X Objective...')
obj = debye_wolf.Objective(WAVELENGTH, FIELD_INDEX, FOCAL_LENGTH, PUPIL_DIAMETER, N, FIELD_WIDTH)
print('...simulated objective with NA={:.3f}'.format(obj._na))

# %% Create linearly polarized pupil

pupil_intensity = gauss2d(N, N // 2)
pupil = debye_wolf.VectorialPupil(pupil_x=pupil_intensity, diameter=PUPIL_DIAMETER)

# %% Calculate field

start = time.time()
field = obj.focus(pupil, Z, FIELD_DEPTH)
elapsed = time.time() - start
print('Simulated {} planes, {:.3f} ms per plane, {:.3f} s total.'.format(Z, (elapsed * 1000) / Z, elapsed))
psf = field.intensity()

# %% Display

plt.figure('Gaussian pupil and field')
ax1 = plt.subplot(1, 3, 1)
plt.title('Pupil')
ax1.imshow(pupil_intensity)
ax1.axis('off')

ax2 = plt.subplot(1, 3, 2)
plt.title('Lateral PSF')
ax2.imshow(psf[:, :, Z // 2])
ax2.axis('off')

ax3 = plt.subplot(1, 3, 3)
plt.title('Axial PSF')
ax3.imshow(np.rot90(psf[:, N // 2, :]))
ax3.axis('off')
