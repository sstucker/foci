# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 01:38:01 2025

@author: sstucker
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from numpy import pi as PI

try:
    from foci import rs
    from foci import util
    from foci import debye_wolf
    from foci import UM, NM, MM
except ModuleNotFoundError:  # If not installed, assume user is running from \examples
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from foci import rs
    from foci import util
    from foci import debye_wolf
    from foci import UM, NM, MM


if __name__ == '__main__':
    
    # Lens parameters
    OBJ_PUPIL_DIAMETER = 20 * MM
    OBJ_FOCAL_LENGTH = 12.5 * MM  # e.g. Nikon 16X
    FIELD_INDEX = 1.33  # Water
    
    λ = 920 * NM
    
    # Simulation parameters
    N = 2048  # Planes are N x N for both propagation modes
    Z = 1  # Depths to simulate
    
    
    PUPIL_SIMULATION_DIAMETER = 20 * MM
    FOCAL_FIELD_WIDTH = 512 * UM
    FOCAL_FIELD_DEPTH = 256 * UM
    
    DX_PUPIL = PUPIL_SIMULATION_DIAMETER / N  # mm/pixels
    
    DX_FOCAL = FOCAL_FIELD_WIDTH / N  # mm/pixels
    DZ_FOCAL = FOCAL_FIELD_DEPTH / Z  # mm/pixels
    
    pupil_extent = np.array((-N // 2, N // 2, -N // 2, N // 2)) * DX_PUPIL
    focal_extent = np.array((-N // 2, N // 2, -N // 2, N // 2)) * DX_FOCAL
    
    # %% Generate objective forward model

    print('Precomputing forward model...')
    objective = debye_wolf.Objective(λ, FIELD_INDEX, OBJ_FOCAL_LENGTH, PUPIL_SIMULATION_DIAMETER, N, FOCAL_FIELD_WIDTH, z=Z, field_depth=FOCAL_FIELD_DEPTH,  multiprocessing=False, precision='complex64')
    print('...simulated objective with NA={:.3f}'.format(objective._na))
    
    # %% Calculate pupil
    
    start = time.time()
    u0 = rs.Field(N, PUPIL_SIMULATION_DIAMETER, objective.wavelength, precision='complex64')
    u0 = u0.init_gaussian(5.0 * MM).mask_circle_aper(2.0 * MM)  # Initial gaussian beam with mask
    u1 = u0.propagate(500 * MM)
    u2 = u1.mask_focus_elliptical(np.inf, 75 * MM).propagate(75 * MM + 200 * MM).mask_focus(200 * MM).propagate(200 * MM)
    pupil = debye_wolf.VectorialPupil(u2._u, diameter=PUPIL_SIMULATION_DIAMETER)
    print('Simulated pupil, {:.3f} s total.'.format(time.time() - start))

    start = time.time()
    field = objective.focus(pupil)
    
    print('Simulated 1 vectorial PSF, {:.3f} s total.'.format(time.time() - start))
    
    # %% Display
    
    plt.figure('Pupil formation')
    ax = plt.subplot(2, 3, 1)
    ax.imshow(u0.intensity(), cmap='gray', extent=pupil_extent)
    ax.axis('off')
    ax.set_title('Incident beam')
    ax = plt.subplot(2, 3, 2)
    ax.imshow(u2.intensity(), cmap='gray', extent=pupil_extent)
    ax.axis('off')
    ax.set_title('Pupil')
    ax = plt.subplot(2, 3, 3)
    ax.imshow(field.intensity(), cmap='gray', extent=focal_extent)
    ax.axis('off')
    ax.set_title('Field')
    
    x_pupil = np.linspace(-N // 2, N // 2, N) * DX_PUPIL / MM
    x_field = np.linspace(-N // 2, N // 2, N) * DX_FOCAL / UM
    
    ax = plt.subplot(2, 3, 4)
    ax.plot(x_pupil, u1.intensity()[N // 2, :], '-k', linewidth=1.0)
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('mm')
    ax = plt.subplot(2, 3, 5)
    ax.plot(x_pupil, u2.intensity()[:, N // 2], '-k', linewidth=1.0)
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('mm')
    ax = plt.subplot(2, 3, 6)
    ax.plot(x_field, field.intensity()[N // 2, :], '-k', linewidth=1.0)
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('um')
    
    plt.tight_layout()
    plt.savefig('aper-cylinder-hybrid-example-psf')
            