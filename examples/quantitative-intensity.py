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
    from foci import debye_wolf
    from foci import UM, NM, MM, M
except ModuleNotFoundError:  # If not installed, assume user is running from \examples
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from foci import debye_wolf
    from foci import UM, NM, MM, M
    

# def get_fwhm(x):
#     x = np.abs(x)
#     if x[0] != np.max(x):
#         raise ValueError('x must be the right-hand side of a PSF beginning at its maximum!')
#     normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
#     i = np.argmin(np.abs(normalized - 0.5))
#     return 2 * i


def get_fwhm(x):
    x = np.abs(x)
    if x[0] != np.max(x):
        raise ValueError('x must be the right-hand side of a PSF beginning at its maximum!')
    n = (x - x.min()) / (x.max() - x.min())
    idx = np.flatnonzero(n < 0.5)
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    f = (0.5 - n[i-1]) / (n[i] - n[i-1]) if n[i] != n[i-1] else 0
    return 2 * (i - 1 + f)


def theoretical_lateral_fwhm(NA, wavelength):
    return (0.515 * wavelength) / NA


def theoretical_axial_fwhm(NA, wavelength):
    return (2.34 * wavelength) / NA**2
    
    
def theoretical_peak(NA, wavelength):
    return (2 * PI * NA**2) / wavelength**2


# %% Generate objective forward model

# Lens parameters
PUPIL_DIAMETER = 8 * MM
WAVELENGTH = 920 * NM

# Field parameters
FIELD_WIDTH = 8 * UM
FIELD_DEPTH = 36 * UM
FIELD_INDEX = 1.33  # Water

# Simulation resolution
N = 128
Z = 128

dx = FIELD_WIDTH / N
dz = FIELD_DEPTH / Z

# %% Create linearly polarized pupil

pupil = debye_wolf.VectorialPupil(pupil_x=np.ones((N, N)), diameter=PUPIL_DIAMETER)
pupil.normalize_power()

# %% Propagate for various objectives

objective_focal_lengths = np.logspace(np.log10(1.6 * MM), np.log10(18 * MM), num=32)

nas = []

intensity_distributions = []

simulated_peaks = []
theoretical_peaks = []

simulated_lateral_fwhms = []
simulated_axial_fwhms = []
theoretical_lateral_fwhms = []
theoretical_axial_fwhms = []

simulated_powers = []

for obj_f in objective_focal_lengths:
    print('Preparing forward model of objective with focal length {:.2f} mm...'.format(obj_f / MM))
    obj = debye_wolf.Objective(WAVELENGTH, FIELD_INDEX, obj_f, PUPIL_DIAMETER, N, FIELD_WIDTH, z=Z, field_depth=FIELD_DEPTH)
    print('...simulated objective with NA={:.3f}'.format(obj._na))
    nas.append(obj._na)
    theoretical_peaks.append(theoretical_peak(obj._na, WAVELENGTH))
    start = time.time()
    field = obj.focus(pupil)
    elapsed = time.time() - start
    print('Propagated to focus, {:.3f} s elapsed.'.format(elapsed))
    # field.save_vectorial('vectorial_gaussian_NA' + str(obj._na)[0:5])
    I = field.intensity()
    simulated_lateral_fwhms.append(get_fwhm(I[N // 2:, N // 2, Z // 2]) * dx)
    simulated_axial_fwhms.append(get_fwhm(I[N // 2, N // 2, Z // 2:]) * dz)
    theoretical_lateral_fwhms.append(theoretical_lateral_fwhm(obj._na, WAVELENGTH))
    theoretical_axial_fwhms.append(theoretical_axial_fwhm(obj._na, WAVELENGTH))
    simulated_peaks.append(I.max())
    intensity_distributions.append(I)
    simulated_powers.append(field.power())

# %% Display

x = np.arange(N // 2) * (FIELD_WIDTH / N)

plt.close('all')

plt.figure('Theory vs. simulation', figsize=(4, 6))

plt.subplot(3, 1, 1)
plt.plot(nas, np.array(theoretical_peaks) / 1e12, '-k', linewidth=1)
plt.scatter(nas, np.array(simulated_peaks) / 1e12, color='black', marker='+')
plt.ylabel('Intensity (TW/m²)')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.subplot(3, 1, 2)
plt.plot(nas, np.array(theoretical_axial_fwhms) / UM, color='darkblue', linewidth=1)
plt.scatter(nas, np.array(simulated_axial_fwhms) / UM, color='darkblue', marker='+', label='Axial')
plt.plot(nas, np.array(theoretical_lateral_fwhms) / UM, color='darkred', linewidth=1)
plt.scatter(nas, np.array(simulated_lateral_fwhms) / UM, color='darkred', marker='+', label='Lateral')
plt.ylabel('FWHM (μm)')
plt.xlabel('Numerical aperture')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.savefig('quantitative-intensity-test')

plt.figure('Integrated power vs NA')
plt.plot(nas, np.ones(len(nas)), '-k', linewidth=1)
plt.scatter(nas, simulated_powers, color='black', marker='+')
plt.ylabel('Power (W)')
plt.xlabel('Numerical aperture')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)