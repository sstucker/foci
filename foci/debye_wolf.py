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
import warnings


MM = 1000.0
UM = 1.0
NM = 1.0 / 1000.0


class FociWarning(UserWarning):
    pass


class VectorialPupil(object):
    
    def __init__(self, pupil_x, pupil_y=None, diameter=None, dtype=np.complex128):
        if pupil_x.shape[0] != pupil_x.shape[1]:
            raise ValueError('The pupil must be square!')
        self._shape = (pupil_x.shape[0], pupil_x.shape[0], 2)
        if diameter is None:
            self._clear_aperture = float(pupil_x.shape[0])
        else:
            self._clear_aperture = diameter
        self._x = np.linspace(-self._clear_aperture / 2, self._clear_aperture / 2, self._shape[0])
        self._aperture_mask = np.zeros((self.n, self.n), dtype=int)  # todo other aperture shapes?
        for i, x in enumerate(self._x):
            for j, y in enumerate(self._x):
                if np.sqrt(x**2 + y**2) < self._clear_aperture / 2:
                    self._aperture_mask[i, j] = 1.0
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
        return self._pupil[:, :, 0] * self._aperture_mask
    
    @property
    def y(self) -> np.ndarray:
        """Returns y-polarized component of pupil field."""
        return self._pupil[:, :, 1] * self._aperture_mask
    
    def set_x(self, pupil_x: np.ndarray):
        self._pupil[:, :, 0] = pupil_x * self._aperture_mask
    
    def set_y(self, pupil_y: np.ndarray):
        self._pupil[:, :, 1] = pupil_y * self._aperture_mask
    
    def modulus(self) -> np.ndarray:
        return np.sqrt(self.x**2 + self.y**2)
    
    def display(self, downsample: int=None, mask_threshold: float=1E-3):
        """
        Display the pupil.

        Parameters
        ----------
        downsample : int, optional
            The downsampled pupil used to draw polarization arrows. If None (default), the pupil polarization is displayed downsampled by factor of 16..
        mask_threshold : float, optional
            The magnitude below which the pupil magnitude is not displayed (considered negligible). The default is 1E-2.
        """
        if downsample is None:
            downsample = self.n // 16
        
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
        ax1.quiver(X[mask], Y[mask], dx * downsample / 6 * u[mask] / a[mask], dx * downsample / 6 * v[mask] / a[mask], color='black', alpha=0.6)
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
            tilt = 180/PI * dp / 2 + 45  # degrees
            h = 1
            w = np.sin(dp)
            ax2.add_patch(Ellipse((_x, _y), h * dx * downsample / 1.5, w * dx * downsample / 1.5, angle=tilt, facecolor='none', edgecolor='black', linewidth=0.2))
            if dp > mask_threshold:
                if _ku < _kv: # If left-handed
                    marker='$l$'
                else:
                    marker='$r$'
                ax2.scatter([_x], [_y], s=8, color='black', marker=marker)
        ax2.set_aspect('equal')
        ax2.set_xlim(-self._clear_aperture / 1.9, self._clear_aperture / 1.9)
        ax2.set_ylim(-self._clear_aperture / 1.9, self._clear_aperture / 1.9)
        cbar = plt.colorbar()
        cbar.set_ticks([-2*PI, -PI, 0, PI, 2*PI])
        cbar.set_ticklabels(['-2π', '-π', '0', 'π', '2π'])
        plt.tight_layout()


# Utility functions


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
    shifted = pupil.x * np.exp(1j * dphase)
    pupil.set_x(shifted)
    if np.max(pupil.y.real) == 0 or np.max(pupil.x.real) == 0:
        warnings.warn('Only one component of the pupil is nonzero: the resulting pupil is not elliptical!', FociWarning)
    return pupil


def vortical_polarize(pupil: VectorialPupil, angle: float=0) -> VectorialPupil:
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
    p = pupil.modulus()
    pupil.set_x(np.cos(phi) * p)
    pupil.set_y(np.sin(phi) * p)
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




# Lens parameters
PUPIL_DIAMETER = 21 * MM
FOCAL_LENGTH = 8 * MM
WAVELENGTH = 750 * NM 

# Field parameters
FOV_WIDTH = 10 * UM
FOV_DEPTH = 100 * UM
FIELD_INDEX = 1.33  # Water

# Simulation resolution
N = 256
Z = 128

pupil = VectorialPupil(ring_beam(N, N // 3, 1), diameter=PUPIL_DIAMETER)
# pupil = VectorialPupil(np.zeros((N, N)), pupil_y=np.ones((N, N)), diameter=PUPIL_DIAMETER)
# pupil = VectorialPupil(np.ones((N, N)), pupil_y=np.zeros((N, N)), diameter=PUPIL_DIAMETER)
# pupil = VectorialPupil(np.ones((N, N)), pupil_y=np.ones((N, N)), diameter=PUPIL_DIAMETER)
# pupil.display()

pupil = vortical_polarize(pupil, angle=PI/2)
# pupil = elliptical_polarize(pupil, dphase=PI/2)

pupil.display()

# %%


class Objective(object):
    
    def __init__(
            self,
            wavelength: float,
            sample_index: float,
            focal_length: float,
            pupil_diameter: int,
            pupil_n: int,
            field_diameter: float,
            precision: str = 'complex128'
         ):
        
        if precision in ['single', 'float', 'float32', 'complex64']:
            self._precision = 'complex64'
        else:
            self._precision = 'complex128'
        
        self._wavelength = wavelength
        self._index = sample_index
        self._focal_length = focal_length
        self._pupil_diameter = pupil_diameter        
        self._N = pupil_n  # Pupil and focal field planar sampling
        self._field_diameter = field_diameter
        
        angle_of_convergence = np.arctan(pupil_diameter / (2 * focal_length))
        self._na = self._index * np.sin(angle_of_convergence)
        
        self._angle_of_convergence = np.arctan(self._pupil_diameter / (2 * self._focal_length))
        self._px_per_m = self._N / self._field_diameter
        
        # Spatial frequency unit and angular bandwidth
        k = self._index * 2 * PI / WAVELENGTH  # Wavenumber
        dk = k / (2 * PI) / self._px_per_m
        k_bandwidth = dk * np.sin(self._angle_of_convergence)
        
        # k-space coordinates of pupil plane
        kx = np.linspace(-k_bandwidth, k_bandwidth, self._N)
        kxx, kyy = np.meshgrid(kx, kx)  # Cartesian grid
        krxy = np.sqrt(kxx**2 + kyy**2)  # Radial k-space grid
        
        # k-space coordinates of z (distance to spherical cap)
        z = dk**2 - krxy**2
        self._kzxy = np.sqrt(np.where(z < 0, 0, z))  # Floor to zero
        
        thetaxy = np.arctan2(krxy, self._kzxy)  # angles made with optical axis
        phixy = np.arctan2(kyy, kxx)  # angles made with a plane transverse the optical axis
        
        self._theta_mask = thetaxy < self._angle_of_convergence  # Circular pupil mask
        
        # Operators from pupil function to spherical k-space (as in [1])
        self._Gx = np.sqrt(np.cos(thetaxy)) *\
            (np.cos(thetaxy) + (1 - np.cos(thetaxy)) * np.sin(phixy)**2) / np.cos(thetaxy)
            
        self._Gy = np.sqrt(np.cos(thetaxy)) *\
            ((np.cos(thetaxy) - 1) * np.cos(phixy) * np.sin(phixy)) / np.cos(thetaxy)
            
        self._Gz = -np.sqrt(np.cos(thetaxy)) *\
            np.sin(thetaxy) * np.cos(phixy) / np.cos(thetaxy)
        
        # Plan CZT
        self._czt = cztw.plan(2, N, w0=-k_bandwidth, w1=k_bandwidth, precision=self._precision)
    
    def _run(self, pupil: VectorialPupil, dz: float = 0.0) -> (np.ndarray, np.ndarray, np.ndarray):
        # Each depth experiences additional free space propagation
        propagation_tf = np.exp(2j * PI * self._kzxy * dz * self._px_per_m)
    
        # x and y components of the pupil
        l_0x = pupil.x * propagation_tf * self._theta_mask
        l_0y = pupil.y * propagation_tf * self._theta_mask
        
        E_Xx = self._czt(l_0x * self._Gx)
        E_Xy = self._czt(l_0x * self._Gy)
        E_Xz = self._czt(-l_0x * self._Gz)
        
        # Eqns 21-24 from [2]
        E_Yx = self._czt(-l_0y * np.rot90(self._Gy))
        E_Yy = self._czt(l_0y * np.rot90(self._Gx))
        E_Yz = self._czt(l_0y * np.rot90(self._Gz))
    
        E_X = E_Xx + E_Yx
        E_Y = E_Xy + E_Yy
        E_Z = E_Xz + E_Yz
        
        I_X = np.abs(E_X)**2
        I_Y = np.abs(E_Y)**2
        I_Z = np.abs(E_Z)**2
    
        return (I_X, I_Y, I_Z)

    
obj = Objective(WAVELENGTH, 1.33, FOCAL_LENGTH, PUPIL_DIAMETER, N, FOV_WIDTH)
print(obj._na)

for z in range(100):
    print(z)
    obj._run(pupil, z)

sys.exit()

angle_of_convergence = np.arctan(PUPIL_DIAMETER / (2 * FOCAL_LENGTH))
numerical_aperture = FIELD_INDEX * np.sin(angle_of_convergence)


# Pixel size
px_per_m = N / FOV_WIDTH
dz = FOV_DEPTH / Z

k = FIELD_INDEX * 2 * PI / WAVELENGTH  # Wavenumber

# Spatial frequency unit and angular bandwidth
dk = k / (2 * PI) / px_per_m
k_bandwidth = dk * np.sin(angle_of_convergence)

# k-space coordinates of pupil plane
kx = np.linspace(-k_bandwidth, k_bandwidth, N)
kxx, kyy = np.meshgrid(kx, kx)  # Cartesian grid
krxy = np.sqrt(kxx**2 + kyy**2)  # Radial k-space grid

# k-space coordinates of z (distance to spherical cap)
z = dk**2 - krxy**2
kzxy = np.sqrt(np.where(z < 0, 0, z))  # Floor to zero

thetaxy = np.arctan2(krxy, kzxy)  # angles made with optical axis
phixy = np.arctan2(kyy, kxx)  # angles made with a plane transverse the optical axis

theta_mask = thetaxy < angle_of_convergence  # Mask of theta within the pupil

# Real operators from pupil function to spherical k-space (as in [1])
Gx = np.sqrt(np.cos(thetaxy)) *\
    (np.cos(thetaxy) + (1 - np.cos(thetaxy)) *
    np.sin(phixy)**2) / np.cos(thetaxy)
    
Gy = np.sqrt(np.cos(thetaxy)) *\
    ((np.cos(thetaxy) - 1) * np.cos(phixy) *
    np.sin(phixy)) / np.cos(thetaxy)
    
Gz = -np.sqrt(np.cos(thetaxy)) *\
    np.sin(thetaxy) * np.cos(phixy) / np.cos(thetaxy)

czt = cztw.plan(2, N, w0=-k_bandwidth, w1=k_bandwidth, precision='complex128')

slices = []

if Z > 1:
    zd = np.linspace(0, FOV_DEPTH / 2, Z // 2 + 1)
    depths = np.concatenate((-zd[:0:-1], zd[:-1]))
else:
    depths = [0]

for dz in depths:
    
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
    
    I_X = np.abs(E_X)**2
    I_Y = np.abs(E_Y)**2
    I_Z = np.abs(E_Z)**2
    
    # plt.figure('Debug: I_X, I_Y, I_Z')
    # plt.subplot(1, 3, 1)
    # plt.imshow(I_X)
    # plt.subplot(1, 3, 2)
    # plt.imshow(I_Y)
    # plt.subplot(1, 3, 3)
    # plt.imshow(I_Z)

    intensity = I_X + I_Y + I_Z

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



