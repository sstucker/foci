# -*- coding: utf-8 -*-
"""
Forward model for focusing of an arbitrary vectorial field by a high NA objective

References
[1] Vishniakou, I. & Seelig, J. D. Differentiable optimization of the Debye-Wolf integral for light shaping and adaptive optics in two-photon microscopy. Opt. Express, OE 31, 9526–9542 (2023).
[2] Boruah, B. R. & Neil, M. A. A. Focal field computation of an arbitrarily polarized beam using fast Fourier transforms. Optics Communications (2009).


Created on Thu Nov  9 12:44:28 2023

@author: sstucker
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
from numpy import pi as PI
import cztw
import multiprocessing


class VectorialPupil(object):
    
    def __init__(self, pupil_x, pupil_y=None, diameter=None, dtype=np.complex128):
        if pupil_x.shape[0] != pupil_x.shape[1]:
            raise ValueError('The pupil must be square!')
        if diameter is None:
            self._clear_aperture = float(pupil_x.shape[0])
        else:
            self._clear_aperture = diameter
        self._shape = (pupil_x.shape[0], pupil_x.shape[0], 2)
        self._x = np.linspace(-self._clear_aperture / 2, self._clear_aperture / 2, self._shape[0])
        self._pupil = np.zeros((self._shape), dtype=dtype)
        self._pupil[:, :, 0] = pupil_x
        if pupil_y is None:  # Default to linear polarization
            self._pupil[:, :, 1] = 0.0 + 0.0j
        else:
            self._pupil[:, :, 1] = pupil_y
        
    def __getitem__(self, slice):
        return self._pupil.__getitem__(slice)
    
    @property
    def n(self):
        return self._shape[0]
    
    @property
    def clear_aperture(self):
        return self._clear_aperture
    
    @property
    def x(self) -> np.ndarray:
        """Returns x-polarized component of pupil field."""
        return self._pupil[:, :, 0]
    
    @property
    def y(self) -> np.ndarray:
        """Returns y-polarized component of pupil field."""
        return self._pupil[:, :, 1]
    
    def set_x(self, pupil_x: np.ndarray):
        self._pupil[:, :, 0] = pupil_x
    
    def set_y(self, pupil_y: np.ndarray):
        self._pupil[:, :, 1] = pupil_y
    
    def modulus(self) -> np.ndarray:
        return np.sqrt(self.x**2 + self.y**2)
    
    def display(self, downsample: int=12, mask_threshold: float=1E-3):
        """
        Display the pupil.

        Parameters
        ----------
        downsample : int, optional
            The downsampled pupil used to draw polarization arrows. The default is 6.
        mask_threshold : float, optional
            The magnitude below which the pupil magnitude is not displayed (considered negligible). The default is 1E-2.
        """
        
        extent = (-self._clear_aperture / 2, self._clear_aperture / 2, -self._clear_aperture / 2, self._clear_aperture / 2)        
        dx = self._x[1] - self._x[0]

        # Quiver plot for polarization direction is downsampled
        X, Y = np.meshgrid(self._x[::downsample], self._x[::downsample])
        u = self.x.real[::downsample, ::downsample]
        v = self.y.real[::downsample, ::downsample]
        a = self.modulus().real[::downsample, ::downsample]

        ku = self.x.imag[::downsample, ::downsample]
        kv = self.y.imag[::downsample, ::downsample]
        ka = self.modulus().imag[::downsample, ::downsample]

        mask_im = (np.abs(self.modulus().real) < mask_threshold)
        for i, x in enumerate(self._x):
            for j, y in enumerate(self._x):
                if np.sqrt(x**2 + y**2) > self._clear_aperture / 2:
                    mask_im[i, j] = True
        mask = ~(mask_im[::downsample, ::downsample])
        
        plt.figure('Pupil', figsize=(12, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax1.add_artist(plt.Circle((0, 0), radius=self._clear_aperture / 2, fill=None, linewidth=3, edgecolor='white'))
        plt.title('Amplitude')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        amp = self.modulus().real
        amp[mask_im] = np.nan
        plt.imshow(amp, extent=extent, cmap='Reds', vmin=0)
        ax1.quiver(X[mask], Y[mask], dx * downsample / 6 * u[mask] / a[mask], dx * downsample / 6 * v[mask] / a[mask], color='black', width=dx * 1E5, alpha=0.6)
        ax1.set_aspect('equal')
        ax1.set_xlim(-self._clear_aperture / 1.9, self._clear_aperture / 1.9)
        ax1.set_ylim(-self._clear_aperture / 1.9, self._clear_aperture / 1.9)
        plt.colorbar()
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.add_artist(plt.Circle((0, 0), radius=self._clear_aperture / 2, fill=None, linewidth=3, edgecolor='white'))
        plt.title('Phase')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ph = self.modulus().imag % PI
        ph[mask_im] = np.nan
        plt.imshow(ph, extent=extent, cmap='Blues', vmin=-2*PI, vmax=2*PI)
        for _x, _y, _ku, _kv, _ka in zip(X[mask].flatten(), Y[mask].flatten(), ku[mask].flatten(), kv[mask].flatten(), ka[mask].flatten()):
            dp = np.abs(_ku - _kv) % PI
            tilt = 180/PI * dp / 2 + 45 # degrees
            h = 1
            w = np.sin(dp)
            ax2.add_patch(Ellipse((_x, _y), h * dx * downsample / 1.5, w * dx * downsample / 1.5, angle=tilt, facecolor='none', edgecolor='black', linewidth=0.2))
            if dp > mask_threshold:
                if _ku < _kv: # If left-handed
                    marker='o'
                else:
                    marker='x'
                ax2.scatter([_x], [_y], s=8, color='black', marker=marker)
        ax2.set_aspect('equal')
        ax2.set_xlim(-self._clear_aperture / 1.9, self._clear_aperture / 1.9)
        ax2.set_ylim(-self._clear_aperture / 1.9, self._clear_aperture / 1.9)
        cbar = plt.colorbar()
        cbar.set_ticks([-2*PI, -PI, 0, PI, 2*PI])
        cbar.set_ticklabels(['-2π', '-π', '0', 'π', '2π'])
        plt.tight_layout()


def elliptical_polarize(pupil: VectorialPupil, dphase: float=PI/2) -> VectorialPupil:
    """
    Applies circular or elliptical polarization to the pupil.

    Parameters
    ----------
    pupil : VectorialPupil
        The pupil instance to polarize..
    dphase : float, optional
        The phase delay applied to the x-component of polarization. The default is PI/2 (circular).

    Returns
    -------
    VectorialPupil
        The polarized pupil instance.

    """
    px = pupil.x
    pupil.set_x(px.real + 1j * (px.imag + dphase))
    return pupil


def vortex_polarize(pupil: VectorialPupil, angle: float=0) -> VectorialPupil:
    """
    Applies vortical polarization to the modulus field of the pupil (existing polarization is lost). 

    Parameters
    ----------
    pupil : VectorialPupil
        The pupil instance to polarize.
    angle : float, optional
        The angle of the vortex in radians. For example, 0 corresponds to radial polarization and π corresponds to azimuthal polarization. The default is 0.

    Returns
    -------
    VectorialPupil
        The polarized pupil instance.

    """
    n = pupil.n
    xx, yy = np.mgrid[-1:1:n*1j, -1:1:n*1j]
    phi = np.arctan2(xx, yy) + angle
    p = pupil.modulus().real
    pupil.set_x(np.cos(phi) * p + pupil.x.imag)
    pupil.set_y(np.sin(phi) * p + pupil.y.imag)
    return pupil


def ring_beam(N, radius, waist):
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


# TODO shift so that mm is 1.0
NM = 10**-9
UM = 10**-6
MM = 10**-3
M = 1.0

# Lens parameters
PUPIL_DIAMETER = 18 * MM
FOCAL_LENGTH = 12.5 * MM
WAVELENGTH = 750 * NM 

# Field parameters
FOV_WIDTH = 10 * UM
FOV_DEPTH = 100 * UM
FIELD_INDEX = 1.33  # Water

# Simulation resolution
N = 256
Z = 256

# pupil = VectorialPupil(ring_beam(N, N // 3, 4), diameter=FOV_WIDTH)
pupil = VectorialPupil(np.ones((N, N)), diameter=FOV_WIDTH)
# pupil.display()


# pupil = VectorialPupil(pupil_x=px, pupil_y=py, diameter=FOV_WIDTH)
pupil = elliptical_polarize(pupil, dphase=PI/2)
# pupil = vortex_polarize(pupil, angle=-PI)

pupil.display()

# %%

angle_of_convergence = np.arctan(PUPIL_DIAMETER / (2 * FOCAL_LENGTH))
numerical_aperture = FIELD_INDEX * np.sin(angle_of_convergence)

# Pixel size
px_per_m = N / FOV_WIDTH
# dz = FOV_DEPTH / Z

k = FIELD_INDEX * 2 * PI / WAVELENGTH  # Wavenumber

# Spatial frequency unit and angular bandwidth
dk = k / (2 * PI) / px_per_m
k_bandwidth = dk * np.sin(angle_of_convergence)

# k-space coordinates of pupil plane
kx = np.linspace(-k_bandwidth, k_bandwidth, N)
kxx, kyy = np.meshgrid(kx, kx)  # Cartesian grid
krxy = np.sqrt(kxx**2 + kyy**2)  # Radial k-space grid

# k-space coordinates of z (distance to spherical cap)
kzxy = np.sqrt(dk**2 - krxy**2)
# kzxy[np.isnan(kzxy)] = 0

thetaxy = np.arctan2(krxy, kzxy)  # angles made with optical axis
phixy = np.arctan2(kyy, kxx)  # angles made with a plane transverse the optical axis

theta_mask = thetaxy < angle_of_convergence  # Mask of theta within the pupil

# Real operators from pupil function to spherical k-space (as in [1])
Gx = np.sqrt(np.cos(thetaxy)) *\
    (np.cos(thetaxy) + (1 - np.cos(thetaxy)) *
    np.sin(phixy) ** 2) / np.cos(thetaxy)
    
Gy = np.sqrt(np.cos(thetaxy)) *\
    ((np.cos(thetaxy) - 1) * np.cos(phixy) *
    np.sin(phixy)) / np.cos(thetaxy)
    
Gz = -np.sqrt(np.cos(thetaxy)) *\
    np.sin(thetaxy) * np.cos(phixy) / np.cos(thetaxy)

czt = cztw.plan(2, N, w0=-k_bandwidth, w1=k_bandwidth, precision='complex64')

slices = []

for dz in np.linspace(-FOV_DEPTH / 2, FOV_DEPTH / 2, Z):
    
    # Each depth experiences additional free space propagation
    propagation_tf = np.exp(2j * PI * kzxy * dz * px_per_m)
    
    # x and y components of the pupil
    l_0x = pupil.x * propagation_tf * theta_mask
    l_0y = pupil.y * propagation_tf * theta_mask
    
    E_Xx = czt(l_0x * Gx)
    E_Xy = czt(l_0x * Gy)
    E_Xz = czt(-l_0x * Gz)
    
    # Eqns 21-24 from [2]
    E_Yx = czt(-l_0y * np.rot90(Gy))
    E_Yy = czt(l_0y * np.rot90(Gx))
    E_Yz = czt(l_0y * np.rot90(Gz))
    
    E_X = E_Xx + E_Yx
    E_Y = E_Xy + E_Yy
    E_Z = E_Xz + E_Yz
    
    intensity = np.abs(E_X)**2 + np.abs(E_Y)**2 + np.abs(E_Z)**2
    
    slices.append(intensity)
    
psf = np.array(slices)

# %%

plt.figure('PSF')
ax1 = plt.subplot(1, 2, 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.imshow(psf[Z // 2, :, :], extent=[-FOV_WIDTH / 2, FOV_WIDTH / 2, -FOV_WIDTH / 2, FOV_WIDTH / 2])
ax2 = plt.subplot(1, 2, 2)
ax2.imshow(psf[:, N // 2, :], extent=[-FOV_WIDTH / 2, FOV_WIDTH / 2, -FOV_DEPTH / 2, FOV_DEPTH / 2])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
plt.xticks([])



