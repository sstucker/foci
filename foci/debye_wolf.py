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

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 8

import cztw
import numpy as np
from numpy import pi as PI
import multiprocessing as mp
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
        
        scaled_aperture = self._clear_aperture / MM
        
        extent = np.array([-scaled_aperture / 2, scaled_aperture / 2, -scaled_aperture / 2, scaled_aperture / 2])
        scaled_x = self._x / MM
        dx = scaled_x[1] - scaled_x[0]
        dx = dx / MM

        # Quiver plot for polarization direction is downsampled
        X, Y = np.meshgrid(scaled_x[::downsample], scaled_x[::downsample])
        u = self.x.real[::downsample, ::downsample]
        v = self.y.real[::downsample, ::downsample]
        a = self.modulus().real[::downsample, ::downsample]

        ku = self.x.imag[::downsample, ::downsample]
        kv = self.y.imag[::downsample, ::downsample]
        ka = self.modulus().imag[::downsample, ::downsample]

        mask_im = (np.abs(self.modulus().real) < mask_threshold)
        for i, x in enumerate(scaled_x):
            for j, y in enumerate(scaled_x):
                if np.sqrt(x**2 + y**2) > scaled_aperture / 2:
                    mask_im[i, j] = True
        mask = ~(mask_im[::downsample, ::downsample])
        
        plt.figure('Pupil', figsize=(8, 3))
        ax1 = plt.subplot(1, 2, 1)
        ax1.add_artist(plt.Circle((0, 0), radius=scaled_aperture / 2, fill=None, linewidth=3, edgecolor='white'))
        plt.title('Amplitude')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        amp = self.modulus().real
        amp[mask_im] = np.nan
        im = plt.imshow(amp, extent=extent, cmap='Reds', vmin=0)
        ax1.quiver(X[mask], Y[mask], dx * downsample / 6 * u[mask] / a[mask], dx * downsample / 6 * v[mask] / a[mask], color='black', alpha=0.6)
        ax1.set_aspect('equal')
        ax1.set_xlim(-scaled_aperture / 1.9, scaled_aperture / 1.9)
        ax1.set_ylim(-scaled_aperture / 1.9, scaled_aperture / 1.9)
        cbar = plt.colorbar(im, fraction=0.02, pad=0.04)
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.add_artist(plt.Circle((0, 0), radius=scaled_aperture / 2, fill=None, linewidth=3, edgecolor='white'))
        plt.title('Phase')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ph = self.modulus().imag % PI
        ph[mask_im] = np.nan
        im = plt.imshow(ph, extent=extent, cmap='Blues', vmin=-2*PI, vmax=2*PI)
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
        ax2.set_xlim(-scaled_aperture / 1.9, scaled_aperture / 1.9)
        ax2.set_ylim(-scaled_aperture / 1.9, scaled_aperture / 1.9)
        ax2.set_yticks([])
        cbar = plt.colorbar(im, fraction=0.02, pad=0.04)
        cbar.set_ticks([-2*PI, -PI, 0, PI, 2*PI])
        cbar.set_ticklabels(['-2π', '-π', '0', 'π', '2π'])
        plt.tight_layout()


class VectorialFocalField(object):
    
    def __init__(
        self,
        shape: tuple,
        field_depth: float,
        field_diameter: float,
        wavelength: float,
        numerical_aperture: float,
        precision: str = 'complex128'
    ):
        self._wavelength = wavelength
        self._numerical_aperture = numerical_aperture
        self._x = np.linspace(-field_diameter / 2, field_diameter / 2, shape[0])
        self._z = np.linspace(-field_depth / 2, field_depth / 2, shape[-1])
            
        self._E = np.empty((*shape, 3), dtype=precision)  # X, Y, and Z field components along last axis
    
    def _assign(self, z: int, e_x: np.ndarray, e_y: np.ndarray, e_z: np.ndarray):
        self._E[:, :, z, 0] = e_x
        self._E[:, :, z, 1] = e_y
        self._E[:, :, z, 2] = e_z
    
    def intensity(self) -> np.ndarray:
        """
        Return the intensity of the focal field (|E_X|^2 + |E_Y|^2+ |E_Z|^2).

        Returns
        -------
        np.ndarray
            Three-dimensional focal field.

        """
        return np.abs(self._E[:, :, :, 0])**2 + np.abs(self._E[:, :, :, 1])**2 + np.abs(self._E[:, :, :, 2])**2
    
    def E_X(self) -> np.ndarray:
        return self._E[:, :, :, 0]
    
    def E_Y(self) -> np.ndarray:
        return self._E[:, :, :, 1]
    
    def E_Z(self) -> np.ndarray:
        return self._E[:, :, :, 2]


# -- Utility functions --------------------------------------------------------


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


# -- Forward model ------------------------------------------------------------


def _simulate_plane(kzxy: np.ndarray, dz, px_per_m, pupil_x, pupil_y, czt, Gx, Gy, Gz):
    # Each depth experiences additional free space propagation
    propagation_tf = np.exp(2j * PI * kzxy * dz * px_per_m)

    # x and y components of the pupil
    l_0x = pupil_x * propagation_tf
    l_0y = pupil_y * propagation_tf
    
    E_Xx = czt(l_0x * Gx)
    E_Xy = czt(l_0x * Gy)
    E_Xz = czt(-l_0x * Gz)
    
    # Eqns 21-24 from [2]
    E_Yx = czt(-l_0y * np.rot90(Gy))
    E_Yy = czt(l_0y * np.rot90(Gx))
    E_Yz = czt(l_0y * np.rot90(Gz))
    
    return (E_Xx + E_Yx, E_Xy + E_Yy, E_Xz + E_Yz)  # Return E_X, E_Y, E_Z


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
        self._m_per_px = self._field_diameter / self._N
        
        # Spatial frequency unit and angular bandwidth
        k = self._index * 2 * PI / self._wavelength  # Wavenumber
        dk = k / (2 * PI) / self._px_per_m
        k_bandwidth = dk * np.sin(self._angle_of_convergence)
        
        # k-space coordinates of pupil plane
        kx = np.linspace(-k_bandwidth, k_bandwidth, self._N)
        kxx, kyy = np.meshgrid(kx, kx)  # Cartesian grid
        krxy = np.sqrt(kxx**2 + kyy**2)  # Radial k-space grid
        
        # k-space coordinates of z (distance to spherical cap)
        z = dk**2 - krxy**2
        self._kzxy = np.sqrt(np.where(z < 0, 0, z))  # Floor to zero
        # self._kzxy = np.sqrt(dk**2 - krxy**2)
        # self._kzxy[np.isnan(self._kzxy)] = 0
        
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
        self._czt = cztw.plan(2, self._N, w0=-k_bandwidth, w1=k_bandwidth, precision=self._precision)
    
    def focus(self, pupil: VectorialPupil, n_depths: np.uint = 1, depth_range: tuple = None, multiprocessing=False) -> VectorialFocalField:
        if n_depths > 1:
            if depth_range is None:  # Default to same sampling as lateral dimension
                depth_range = n_depths * self._m_per_px
            zd = np.linspace(0, depth_range / 2, n_depths // 2 + 1)
            depths = np.concatenate((-zd[:0:-1], zd[:-1]))
        else:
            depths = [0]
        # todo parallelize
        field_shape = (self._N, self._N, n_depths)
        field = VectorialFocalField(field_shape, depth_range, self._field_diameter, self._wavelength, self._na)
        def worker(dz):
            return _simulate_plane(self._kzxy, dz, self._px_per_m, pupil.x, pupil.y, self._czt, self._Gx, self._Gy, self._Gz)
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            results = pool.map_async(worker, depths)
            pool.close()
            pool.join()
        else:
            for z, dz in enumerate(depths):
                e_x, e_y, e_z = worker(dz)
                field._assign(z, e_x, e_y, e_z)
        return field
