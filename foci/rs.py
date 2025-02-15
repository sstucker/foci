# -*- coding: utf-8 -*-
"""

Rayleigh-Sommerfeld Propagation

[1] Brzobohatý et al 2008 https://doi.org/10.1364/OE.16.012688

Created on Thu Jan 23 01:31:27 2025

@author: sstucker

"""

import numpy as np
import pyfftw as fftw
import poppy.zernike
import scipy.ndimage

from foci.util import *

PI = np.pi

MM = 1000.0
UM = 1.0
NM = 1.0 / 1000.0


PLANNER_EFFORT = 'FFTW_MEASURE'


def gaussian(N, waist):
    r = np.linspace(-N / 2, N / 2, N, dtype=np.float64)
    x, y = np.meshgrid(r, r)
    return np.exp(-(x**2 + y**2) / waist**2)


# -- Phase masks for common optical elements ----------------------------------


def axicon(N, L, λ, θ, index, thickness, curvature=0, decenter=(0, 0), tilt=0):
    """Axicon phase mask with variable tip curvature from [1]."""
    dx = L / N
    k = 2 * PI / λ  # Wavenumber
    r = np.linspace(-L / 2, L / 2, N)
    x, y = np.meshgrid(r, r)
    scale_x = np.cos(tilt)
    scale_y = 1
    rr = np.sqrt(((x * scale_x) + decenter[0])**2 + ((y * scale_y) + decenter[1])**2)
    phi = (index - 1.0) * (np.sqrt(rr**2 + curvature**2) + curvature) * np.tan(θ)
    return np.exp(1j * k * phi)


def thin_lens(N, L, λ, fx, fy, decenter=(0, 0)):
    dx = L / N
    k = 2 * PI / λ  # Wavenumber
    r = np.linspace(-L / 2, L / 2, N)
    x, y = np.meshgrid(r, r)
    x += decenter[0]
    y += decenter[1]
    return np.exp(-1j * k * (x**2 / fx + y**2 / fy) / 2)

def tilt(N, L, λ, θ_x, θ_y):
    k = 2 * PI / λ  # Wavenumber
    r = np.linspace(-L / 2, L / 2, N)
    x, y = np.meshgrid(r, r)
    return np.exp(1j * k * (x * np.sin(θ_x) + y * np.sin(θ_y)))


# -- Transfer functions and real space operations------------------------------


def propagate(N, L, λ, z, index=1.0):
    """Returns RS free space propagator of distance `z` in spatial freq domain (0 centered)."""
    dx = L / N  # Spatial sampling interval
    k = 2 * PI / λ  # Wavenumber
    F_r = np.fft.fftfreq(N, d=dx)
    F_r = np.fft.fftshift(F_r)
    F_x, F_y = np.meshgrid(F_r, F_r)
    kz = np.sqrt((k * index)**2 - (2 * PI * F_x)**2 - (2 * PI * F_y)**2 + 0j)
    kz[np.real(kz) < 0] = 0  # Remove evanescent waves
    H = np.exp(1j * kz * z)
    return H

def decenter(U, N, L, dx, dy):
    r = np.linspace(-L / 2, L / 2, N)
    x, y = np.meshgrid(r, r)
    interp_func = scipy.interpolate.RegularGridInterpolator(
        points=(r, r),
        values=U,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )
    shifted = np.stack(((y + dy).ravel(), (x + dx).ravel()), axis=-1)
    return np.reshape(interp_func(shifted), (N, N))


# -----------------------------------------------------------------------------


class Field:
    
    def __init__(self, N, L, λ, u=None, precision='double', _transforms=None):
        if precision in ['double, complex128', '128']:
            self._complex_dtype = 'complex128'
        else:
            self._complex_dtype = 'complex64'
        self._width = L
        self._N = N
        self._r = np.linspace(-self._width / 2, self._width / 2, N)
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
    
    def decenter(self, dx, dy):
        dx_pixel = dx / self._dr
        dy_pixel = dy / self._dr
        decentered_u = scipy.ndimage.shift(self._u, (dx_pixel, dy_pixel))
        return Field(self._N, self._width, self._wavelength, u=decentered_u, precision=self._complex_dtype, _transforms=(self._fft_fwd, self._fft_bwd))
    
    def mask_circle_aper(self, r):
        x, y = np.meshgrid(self._r, self._r)
        mask = np.sqrt(x**2 + y**2) < r
        u = self._u * mask
        return Field(self._N, self._width, self._wavelength, u=u, precision=self._complex_dtype, _transforms=(self._fft_fwd, self._fft_bwd))
        
    def mask_focus(self, f, decenter=(0, 0)):
        """Return field focused by a thin lens of focal length `f`."""
        u = self._u * thin_lens(self._N, self._width, self._wavelength, f, f, decenter=decenter)
        return Field(self._N, self._width, self._wavelength, u=u, precision=self._complex_dtype, _transforms=(self._fft_fwd, self._fft_bwd))
    
    def mask_focus_elliptical(self, fx, fy):
        u = self._u * thin_lens(self._N, self._width, self._wavelength, fx, fy)
        return Field(self._N, self._width, self._wavelength, u=u, precision=self._complex_dtype, _transforms=(self._fft_fwd, self._fft_bwd))
        
    def mask_axicon(self, theta, thickness, curvature=0, index=1.5, decenter=(0, 0), tilt=0):
        u = self._u * axicon(self._N, self._width, self._wavelength, theta, index, thickness, curvature=curvature, decenter=decenter, tilt=tilt)
        return Field(self._N, self._width, self._wavelength, u=u, precision=self._complex_dtype, _transforms=(self._fft_fwd, self._fft_bwd))
    
    def mask_tilt(self, theta_x, theta_y):
        u = self._u * tilt(self._N, self._width, self._wavelength, theta_x, theta_y)
        return Field(self._N, self._width, self._wavelength, u=u, precision=self._complex_dtype, _transforms=(self._fft_fwd, self._fft_bwd)) 
