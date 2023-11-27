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
FIELD_WIDTH = 6 * UM
FIELD_DEPTH = 50 * UM
FIELD_INDEX = 1.33  # Water

# Simulation resolution
N = 128
Z = 128

print('Precomputing forward model...')
obj = debye_wolf.Objective(WAVELENGTH, FIELD_INDEX, FOCAL_LENGTH, PUPIL_DIAMETER, N, FIELD_WIDTH)
print('...simulated objective with NA={:.3f}'.format(obj._na))

# %% Calculate PSFs

waists = np.linspace(1, 14, 10)[::-1]  # Gaussian sigma of the annulus in px
psfs = []
pupils = []

for waist in waists:
    # Generate linearly-polarized annular pupil
    pupil = debye_wolf.VectorialPupil(ring_beam(N, N // 4, waist), diameter=PUPIL_DIAMETER)
    start = time.time()
    field = obj.focus(pupil, Z, FIELD_DEPTH)
    elapsed = time.time() - start
    print('Simulated {} planes, {:.3f} ms per plane, {:.3f} s total.'.format(Z, (elapsed * 1000) / Z, elapsed))
    psf = field.intensity()
    psfs.append(psf)
    pupil_intensity = pupil.modulus().real
    pupil_intensity[~pupil.mask] = None
    pupils.append(pupil_intensity)

# %% Display
    
f = plt.figure('Bessel-Gauss PSFs', figsize=(8, 5))
for i, (psf, pupil) in enumerate(zip(psfs, pupils)):
    ax1 = plt.subplot(2, len(waists), i + 1)
    ax1.imshow(pupil, cmap='gray')
    ax1.axis('off')
    ax1.margins(x=0, y=0)
    ax2 = plt.subplot(2, len(waists), i + len(waists) + 1)
    ax2.imshow(np.rot90(psf[N // 2, :, :]), cmap='gray', extent=(-FIELD_WIDTH / 2, FIELD_WIDTH / 2, -FIELD_DEPTH / 2, FIELD_DEPTH / 2))
    ax2.axis('off')
    ax2.margins(x=0, y=0)
plt.tight_layout()
plt.savefig('bessel-gauss-example.png')
