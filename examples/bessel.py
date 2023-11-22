# Simulate Bessel-Gauss beams with various axial extents and depth profiles

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import time


try:
    from foci import debye_wolf
    from foci import UM, NM, MM
except ModuleNotFoundError:  # If not installed, assume user is running from \examples
    sys.path.insert(0, os.path.abspath('..'))
    sys.path.insert(0, os.path.join(os.path.abspath('..'), 'foci'))
    from foci import debye_wolf
    from foci import UM, NM, MM


def ring_beam(N, radius, waist):
    """Generate a Gaussian ring on an N x N pupil."""
    r = np.linspace(-N / 2, N / 2, N, dtype=np.float64)
    xc = np.zeros(N)
    xc[N // 2 - int(radius)] = 1.0
    xc[N // 2 + int(radius)] = 1.0
    kernel = np.exp(-(r / waist)**2 / 2)
    xc = np.convolve(xc, kernel)[N // 2:-N // 2 + 1].astype(np.float64)
    field = np.zeros((N, N), dtype=np.complex128)
    for i, x in enumerate(r):
        for j, y in enumerate(r):
            field[i, j] = np.interp(np.sqrt(x**2 + y**2), r, xc) + 0j
    return field + 0j


# %% Generate objective forward model

# Lens parameters
PUPIL_DIAMETER = 21 * MM
FOCAL_LENGTH = 8 * MM
WAVELENGTH = 750 * NM

# Field parameters
FIELD_WIDTH = 10 * UM
FIELD_DEPTH = 24 * UM
FIELD_INDEX = 1.33  # Water

# Simulation resolution
N = 256
Z = 256

# Generate linearly-polarized annular pupil
pupil = debye_wolf.VectorialPupil(ring_beam(N, N // 3, 6), diameter=PUPIL_DIAMETER)

print('Precomputing forward model...')
obj = debye_wolf.Objective(WAVELENGTH, FIELD_INDEX, FOCAL_LENGTH, PUPIL_DIAMETER, N, FIELD_WIDTH)
print('...simulated objective with NA={:.3f}'.format(obj._na))

start = time.time()

field = obj.focus(pupil, Z, FIELD_DEPTH)

elapsed = time.time() - start
print('Simulated {} planes, {:.3f} ms per plane'.format(Z, (elapsed * 1000) / Z))

psf = field.intensity()

# %%

plt.close('PSF')
plt.figure('PSF')
ax1 = plt.subplot(1, 2, 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.imshow(psf[:, :, Z // 2], extent=[-FIELD_WIDTH / 2, FIELD_WIDTH / 2, -FIELD_WIDTH / 2, FIELD_WIDTH / 2])
ax2 = plt.subplot(1, 2, 2)
ax2.imshow(np.rot90(psf[N // 2, :, :]), extent=[-FIELD_WIDTH / 2, FIELD_WIDTH / 2, -FIELD_DEPTH / 2, FIELD_DEPTH / 2])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
plt.xticks([])







