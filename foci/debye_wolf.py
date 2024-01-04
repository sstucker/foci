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
rcParams['font.size'] = 12

import time
import os

import cztw
import numpy as np
from numpy import pi as PI
import multiprocessing as mp
import warnings
import ctypes

mp = mp.get_context('spawn')

MM = 1000.0
UM = 1.0
NM = 1.0 / 1000.0


ctype = {
    'complex64': np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags='C_CONTIGUOUS'),
    'complex128': np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1, flags='C_CONTIGUOUS')
}


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
    def mask(self):
        return self._aperture_mask
    
    @property
    def x(self) -> np.ndarray:
        """Returns x-polarized component of pupil field."""
        return self._pupil[:, :, 0] * self._aperture_mask
    
    @property
    def y(self) -> np.ndarray:
        """Returns y-polarized component of pupil field."""
        return self._pupil[:, :, 1] * self._aperture_mask
    
    @property
    def xy(self) -> np.ndarray:
        """Returns both components of the pupil field stacked along the last axis."""
        return np.stack((self.x, self.y), axis=-1)
    
    @property
    def x_amplitude(self) -> np.ndarray:
        """Returns x-polarized amplitude of pupil field."""
        return np.abs(self._pupil[:, :, 0] * np.sign(self._pupil[:, :, 0])) * self._aperture_mask
    
    @property
    def y_amplitude(self) -> np.ndarray:
        """Returns y-polarized amplitude of pupil field."""
        return np.abs(self._pupil[:, :, 1] * np.sign(self._pupil[:, :, 1])) * self._aperture_mask

    @property
    def x_phase(self) -> np.ndarray:
        """Returns the phase of the x-polarized component of the pupil field."""
        return np.angle(self._pupil[:, :, 0]) * self._aperture_mask
    
    @property
    def y_phase(self) -> np.ndarray:
        """Returns the phase of the y-polarized component of the pupil field."""
        return np.angle(self._pupil[:, :, 1]) * self._aperture_mask
    
    def set_x(self, pupil_x: np.ndarray):
        self._pupil[:, :, 0] = pupil_x * self._aperture_mask
    
    def set_y(self, pupil_y: np.ndarray):
        self._pupil[:, :, 1] = pupil_y * self._aperture_mask
    
    def intensity(self) -> np.ndarray:
        return np.real(self.x * np.conj(self.x) + self.y * np.conj(self.y))
    
    def display(self, downsample: int=None, mask_threshold: float=1E-3, display_phase=True, cmap_amp='Reds', cmap_phase='Blues'):
        
        if downsample is None:
            downsample = self.n // 16
        
        scaled_aperture = self._clear_aperture / MM
        
        extent = np.array([-scaled_aperture / 2, scaled_aperture / 2, -scaled_aperture / 2, scaled_aperture / 2])
        scaled_x = self._x / MM
        dx = scaled_x[1] - scaled_x[0]
        dx = dx / MM

        # Quiver plot for polarization direction is downsampled
        X, Y = np.meshgrid(scaled_x[::downsample], scaled_x[::downsample])
        u = self.x_amplitude[::downsample, ::downsample]
        v = self.y_amplitude[::downsample, ::downsample]
        a = self.intensity()[::downsample, ::downsample]
        ku = self.x_phase[::downsample, ::downsample]
        kv = self.y_phase[::downsample, ::downsample]

        mask_im = (self.intensity() < mask_threshold)
        for i, x in enumerate(scaled_x):
            for j, y in enumerate(scaled_x):
                if np.sqrt(x**2 + y**2) > scaled_aperture / 2:
                    mask_im[i, j] = True
        mask = ~(mask_im[::downsample, ::downsample])
        
        plt.figure('Pupil')
        ax1 = plt.subplot(1, (1, 2)[int(display_phase)], 1)
        ax1.add_artist(plt.Circle((0, 0), radius=scaled_aperture / 2, fill=None, linewidth=3, edgecolor='white'))
        plt.title('Pupil amplitude')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        amp = self.intensity()
        amp[mask_im] = np.nan
        im = plt.imshow(amp, extent=extent, cmap=cmap_amp, vmin=0)
        ax1.quiver(X[mask], Y[mask], dx * downsample / 6 * u[mask], dx * downsample / 6 * v[mask], color='black', alpha=0.6)
        ax1.set_aspect('equal')
        ax1.set_xlim(-scaled_aperture / 1.9, scaled_aperture / 1.9)
        ax1.set_ylim(-scaled_aperture / 1.9, scaled_aperture / 1.9)
        plt.colorbar(im, fraction=0.02, pad=0.04)
        if display_phase:
            pass
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
        self._shape = shape
    
    @property
    def shape(self) -> tuple:
        return self._shape
    
    def _assign(self, z: int, e_x: np.ndarray, e_y: np.ndarray, e_z: np.ndarray):
        self._E[:, :, z, 0] = e_x
        self._E[:, :, z, 1] = e_y
        self._E[:, :, z, 2] = e_z
    
    def intensity(self) -> np.ndarray:
        """
        Return the intensity of the focal field (|E_X|^2 + |E_Y|^2+ |E_Z|^2).
z
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
    Applies vortical polarization to the x-component of the pupil (existing polarization and y-component are lost). 

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
    if np.max(pupil.y) > 0:
        warnings.warn('Y-component lost! vortical_polarize destroys y-component of pupil field.', FociWarning)
    p = pupil.x
    pupil.set_x(np.cos(phi) * p)
    pupil.set_y(np.sin(phi) * p)
    return pupil


# -- Forward model ------------------------------------------------------------


def _worker(done, data, job_queue, result_queue):
    czt = cztw.plan(**data['cztw_plan_args'])
    shape = (data['cztw_plan_args']['N'], data['cztw_plan_args']['N'])
    # print(os.getpid(), 'Planned transform.', flush=True)
    while not done.is_set():
        # print(os.getpid(), 'Waiting for work with plan', czt, flush=True)
        if not job_queue.empty():
            job = job_queue.get()
            i = job['i']
            # print('Job', job['i'], 'received!')
            E_X, E_Y, E_Z = _simulate_plane(job['pupil'], job['dz'], data['kzxy'], data['px_per_m'], czt, data['Gx'], data['Gy'], data['Gz'])
            result_queue.put(dict(i=i, e_x=E_X, e_y=E_Y, e_z=E_Z))
            # print(os.getpid(), 'finished job', i, flush=True)

def _simulate_plane(pupil: np.ndarray, dz: float, kzxy: np.ndarray, px_per_m, czt, Gx, Gy, Gz):
    # Each depth experiences additional free space propagation
    propagation_tf = np.exp(2j * PI * kzxy * dz * px_per_m)

    # x and y components of the pupil
    l_0x = pupil[:, :, 0] * propagation_tf
    l_0y = pupil[:, :, 1] * propagation_tf
    
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
            precision: str = 'complex128',
            multiprocessing: bool = False
        ):
        
        if precision in ['single', 'float', 'float32', 'complex64']:
            self._precision = 'complex64'
        else:
            self._precision = 'complex128'
        self._parallelized = bool(multiprocessing)
        self._n_processes = mp.cpu_count()
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
        self._k_bandwidth = dk * np.sin(self._angle_of_convergence)
        
        # k-space coordinates of pupil plane
        kx = np.linspace(-self._k_bandwidth, self._k_bandwidth, self._N)
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
        
        # Await first call of _prepare_resources to spawn threads and plan ffts
        self._ready = False
        self._done = mp.Event()
        self._czt = None
        self._job_queue = None
        self._result_queue = None
        self._worker_pool = []
    
    def _setup(self):
        """The most time-consuming operations carried out the first time an Objective is used to simulate should be here."""
        if self._parallelized:
            self._job_queue = mp.Queue()
            self._result_queue = mp.Queue()
            for i in range(self._n_processes):
                init_data = {
                    'cztw_plan_args': dict(ndim=2, N=self._N, M=None, w0=-self._k_bandwidth, w1=self._k_bandwidth, precision=self._precision),
                    'kzxy': self._kzxy.copy(),
                    'px_per_m': self._px_per_m,
                    'Gx': self._Gx.copy(),
                    'Gy': self._Gy.copy(),
                    'Gz': self._Gz.copy()
                }
                self._worker_pool.append(mp.Process(target=_worker, args=(self._done, init_data, self._job_queue, self._result_queue)))
            for process in self._worker_pool:
                # print('Starting process...')
                process.start()
        else:
            self._czt = cztw.plan(2, self._N, w0=-self._k_bandwidth, w1=self._k_bandwidth, precision=self._precision)
        self._ready = True

    def focus(self, pupil: VectorialPupil, n_depths: np.uint = 1, depth_range: tuple = None) -> VectorialFocalField:
        if not self._ready:
            self._setup()
        if n_depths > 1:
            if depth_range is None:  # Default to same sampling as lateral dimension
                depth_range = n_depths * self._m_per_px
            zd = np.linspace(0, depth_range / 2, n_depths // 2 + 1)
            depths = np.concatenate((-zd[:0:-1], zd[:-1]))
        else:
            depths = [0]
        field_shape = (self._N, self._N, n_depths)
        field = VectorialFocalField(field_shape, depth_range, self._field_diameter, self._wavelength, self._na)
        if self._parallelized and n_depths > 1:
            for i, dz in enumerate(depths):
                self._job_queue.put(dict(pupil=pupil.xy, dz=dz, i=i))
            finished_jobs = 0
            while finished_jobs < len(depths):
                if not self._result_queue.empty():
                    result = self._result_queue.get()
                    field._assign(result['i'], result['e_x'], result['e_y'], result['e_z'])
                    finished_jobs += 1
                    # print('Finished', finished_jobs, 'of', len(depths), 'jobs')
            print(self._result_queue.qsize(), 'jobs left in queue (should be zero!)')
        else:
            for i, dz in enumerate(depths):
                e_x, e_y, e_z = _simulate_plane(pupil.xy, dz, self._kzxy, self._px_per_m, self._czt, self._Gx, self._Gy, self._Gz)
                field._assign(i, e_x, e_y, e_z)
        return field

    def __del__(self):
        for process in self._worker_pool:
            process.join()
        self._done.set()
