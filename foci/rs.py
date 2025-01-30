# -*- coding: utf-8 -*-
"""

Rayleigh-Sommerfeld Propagation

[1] Brzobohatý et al 2008 https://doi.org/10.1364/OE.16.012688

Created on Thu Jan 23 01:31:27 2025

@author: sstucker

"""

import numpy as np
import pyfftw as fftw

from foci.util import *

PI = np.pi

MM = 1000.0
UM = 1.0
NM = 1.0 / 1000.0


PLANNER_EFFORT = 'FFTW_PATIENT'


def gaussian(N, waist):
    r = np.linspace(-N / 2, N / 2, N, dtype=np.float64)
    x, y = np.meshgrid(r, r)
    return np.exp(-(x**2 + y**2) / waist**2)


# -- Phase masks for common optical elements ----------------------------------


def axicon(N, L, λ, θ, index, thickness, curvature=0):
    """Axicon phase mask with variable tip curvature from [1]."""
    dx = L / N
    k = 2 * PI / λ  # Wavenumber
    r = np.linspace(-L / 2, L / 2, N)
    x, y = np.meshgrid(r, r)
    rr = np.sqrt(x**2 + y**2)
    return np.exp(1j * k * (index - 1.0) * (np.sqrt(rr**2 + curvature**2) - curvature) * np.tan(θ))

def thin_lens(N, L, λ, f):
    dx = L / N
    k = 2 * PI / λ  # Wavenumber
    if dx < (λ * f) / N:
        print('Warning! Undersampled!')
    r = np.linspace(-L / 2, L / 2, N)
    x, y = np.meshgrid(r, r)
    return np.exp(-1j * k * (x**2 + y**2) / (2 * f))


# -- Transfer functions -------------------------------------------------------


def propagate(N, L, λ, z, index=1.0):
    """Returns RS free space propagator of distance `z` in spatial freq domain (0 centered)."""
    dx = L / N  # Spatial sampling interval
    k = 2 * PI / λ  # Wavenumber
    F_r = np.fft.fftfreq(N, d=dx)
    F_r = np.fft.fftshift(F_r)
    F_x, F_y = np.meshgrid(F_r, F_r)
    kz = np.sqrt((k * index)**2 - (2 * PI * F_x)**2 - (2 * PI * F_y)**2 + 0j)
    # kz[np.real(kz) < 0] = 0  # Remove evanescent waves
    H = np.exp(1j * kz * z)
    return H


# -----------------------------------------------------------------------------


class Field:
    
    def __init__(self, N, L, λ, u=None, precision='double', _transforms=None):
        if precision in ['double, complex128', '128']:
            self._complex_dtype = 'complex128'
        else:
            self._complex_dtype = 'complex64'
        self._width = L
        self._N = N
        self._dr = L / N  # unit / px
        self._wavelength = λ
        self._shape = (N, N)
        if u is None:
            self._u = np.zeros(self._shape, dtype=self._complex_dtype)
        else:
            if u.shape == self._shape:
                self._u = u.astype(self._complex_dtype)
            else:
                raise ValueError('`u` must be shape {}'.format(self._shape))
        if _transforms is None:
            self._fft_fwd = None
            self._fft_bwd = None
        else:
            self._fft_fwd = _transforms[0]
            self._fft_bwd = _transforms[1]
        self._x = None
        self._X = None
                
    @property
    def u(self) -> np.ndarray:
        return self._u
        
    def _plan(self):
        import_fftw_wisdom()
        self._x = fftw.empty_aligned(self._shape, dtype=self._complex_dtype)
        self._X = fftw.empty_aligned(self._shape, dtype=self._complex_dtype)
        self._fft_fwd = fftw.builders.fft2(self._x, overwrite_input=True, avoid_copy=True, threads=1, planner_effort=PLANNER_EFFORT, auto_align_input=True)
        self._fft_bwd = fftw.builders.ifft2(self._X, overwrite_input=True, avoid_copy=True, threads=1, planner_effort=PLANNER_EFFORT, auto_align_input=True)
        export_fftw_wisdom()
    
    def init_gaussian(self, w0):
        self._u[:] = gaussian(self._N, w0 / self._dr)
        
    def init_plane(self):
        self._u[:] = 1.0
    
    def propagate(self, distance):
        """Return new `Field` following free-space propagation of `distance` along the optical axis."""
        if self._fft_fwd is None:
            self._plan()
        else:
            self._x = fftw.empty_aligned(self._shape, dtype=self._complex_dtype)
            self._X = fftw.empty_aligned(self._shape, dtype=self._complex_dtype)
        H = propagate(self._N, self._width, self._wavelength, distance)
        self._x[:] = self._u
        self._X = self._fft_fwd(self._x)
        self._X = self._X * np.fft.fftshift(H)  # Apply transfer function
        self._x = self._fft_bwd(self._X)
        return Field(self._N, self._width, self._wavelength, u=self._x.copy(), precision=self._complex_dtype, _transforms=(self._fft_fwd, self._fft_bwd))
    
    def mask_focus(self, f):
        """Return field focused by a thin lens of focal length `f`."""
        u = self._u * thin_lens(self._N, self._width, self._wavelength, f)
        return Field(self._N, self._width, self._wavelength, u=u, precision=self._complex_dtype, _transforms=(self._fft_fwd, self._fft_bwd))
        
    def mask_axicon(self, theta, thickness, curvature=0, index=1.5):
        u = self._u * axicon(self._N, self._width, self._wavelength, theta, index, thickness, curvature=curvature)
        return Field(self._N, self._width, self._wavelength, u=u, precision=self._complex_dtype, _transforms=(self._fft_fwd, self._fft_bwd))

    def __del__(self):
        export_fftw_wisdom()
