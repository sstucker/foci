# -*- coding: utf-8 -*-
"""

Demonstrates that the simulation matches theoretical results for Gaussian
pupils with unit power as NA varies

Created on Tue Dec 19 11:12:12 2023

@author: tuckes06
"""

import numpy as np
from numpy import pi as PI
import matplotlib.pyplot as plt
import os
import sys
import time

try:
    from foci import util
    from foci import debye_wolf
    from foci import UM, NM, MM
except ModuleNotFoundError:  # If not installed, assume user is running from \examples
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from foci import util
    from foci import debye_wolf
    from foci import UM, NM, MM
    

def get_fwhm(x):
    x = np.abs(x)
    if x[0] != np.max(x):
        raise ValueError('x must be the right-hand side of a PSF beginning at its maximum!')
    normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
    i = np.argmin(np.abs(normalized - 0.5))
    return 2 * i


def theoretical_fwhm(NA, wavelength):
    return (0.515 * wavelength) / NA
    
    
def theoretical_peak(NA, wavelength):
    return (2 * PI * NA**2) / wavelength**2


# %% Generate objective forward model

# Lens parameters
PUPIL_DIAMETER = 8 * MM
WAVELENGTH = 920 * NM

# Field parameters
FIELD_WIDTH = 22 * UM
FIELD_INDEX = 1.0  # Water

# Simulation resolution
N = 256

# %% Create linearly polarized pupil

pupil = debye_wolf.VectorialPupil(pupil_x=np.ones((N, N)), diameter=PUPIL_DIAMETER)
pupil.normalize_power()

# %% Propagate for various objectives

objective_focal_lengths = np.logspace(np.log10(2 * MM), np.log10(80 * MM), num=24)
nas = []
intensity_distributions = []
theoretical_peaks = []
simulated_peaks = []
simulation_power = []
simulated_fwhms = []
theoretical_fwhms = []

for obj_f in objective_focal_lengths:
    print('Preparing forward model of objective with focal length {:.2f} mm...'.format(obj_f / MM))
    obj = debye_wolf.Objective(WAVELENGTH, FIELD_INDEX, obj_f, PUPIL_DIAMETER, N, FIELD_WIDTH)
    print('...simulated objective with NA={:.3f}'.format(obj._na))
    nas.append(obj._na)
    theoretical_peaks.append(theoretical_peak(obj._na, WAVELENGTH))
    start = time.time()
    field = obj.focus(pupil)
    elapsed = time.time() - start
    print('Propagated to focus, {:.3f} s elapsed.'.format(elapsed))
    I = field.intensity()
    simulated_fwhms.append(get_fwhm(I[N // 2:, N // 2]) * (FIELD_WIDTH / N))
    theoretical_fwhms.append(theoretical_fwhm(obj._na, WAVELENGTH))
    intensity_distributions.append(I)
    
    simulated_peaks.append(I.max())
    simulation_power.append(np.sum(I) * (PUPIL_DIAMETER / N)**2)

# %% Display

x = np.arange(N // 2) * (FIELD_WIDTH / N)

plt.close('all')

plt.figure('Theory vs. simulation', figsize=(4, 6))

plt.subplot(2, 1, 1)
plt.plot(nas, theoretical_peaks, '-k', linewidth=1)
plt.scatter(nas, simulated_peaks, color='black', marker='+')
plt.ylabel('Intensity (W/m²)')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.subplot(2, 1, 2)
plt.plot(nas, theoretical_fwhms, '-k', linewidth=1)
plt.scatter(nas, simulated_fwhms, color='black', marker='+')
plt.ylabel('FWHM (μm)')
plt.xlabel('Numerical aperture')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.savefig('quantitative-intensity-test')
