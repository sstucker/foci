# -*- coding: utf-8 -*-
"""
Forward model for focusing of an arbitrary vectorial field by a high NA objective

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

from foci import cztw
import numpy as np
from numpy import pi as PI
from multiprocessing import shared_memory as sm
import multiprocessing as mp
import warnings


mp = mp.get_context('spawn')


MM = 1000.0
UM = 1.0
NM = 1.0 / 1000.0


type_bytes = {
    'complex64': 8,
    'complex128': 16
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
    
    @property
    def n(self):
        return self._shape[0]
    
    @property
    def shape(self):
        return self._shape
    
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
        return self._pupil[:, :, 0] * self._aperture_mask
    
    @property
    def y_amplitude(self) -> np.ndarray:
        """Returns y-polarized amplitude of pupil field."""
        return self._pupil[:, :, 1] * self._aperture_mask

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
    
    def display(self, mask_threshold: float=1E-9, display_phase=True, display_polarization=True, polarization_downsample: int=None, cmap_amp='Reds', cmap_phase='Blues', display_colorbar=False):
        
        if polarization_downsample is None:
            downsample = self.n // 16
        else:
            downsample = polarization_downsample

        scaled_aperture = self._clear_aperture / MM
        
        extent = np.array([-scaled_aperture / 2, scaled_aperture / 2, -scaled_aperture / 2, scaled_aperture / 2])
        scaled_x = self._x / MM
        dx = scaled_x[1] - scaled_x[0]
        dx = dx / MM

        amp = self.intensity()

        mask_im = (amp / np.max(amp) < mask_threshold)
        for i, x in enumerate(scaled_x):
            for j, y in enumerate(scaled_x):
                if np.sqrt(x**2 + y**2) > scaled_aperture / 2:
                    mask_im[i, j] = True
        mask = ~(mask_im[::downsample, ::downsample])
        
        plt.figure('Pupil')
        ax1 = plt.subplot(1, (1, 2)[int(display_phase)], 1)
        ax1.add_artist(plt.Circle((0, 0), radius=scaled_aperture / 2, fill=None, linewidth=3, edgecolor='white'))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        amp[mask_im] = np.nan
        im = plt.imshow(amp, extent=extent, cmap=cmap_amp, vmin=0)
        if display_polarization:
            plt.title('Pupil amplitude & polarization')
            X, Y = np.meshgrid(scaled_x[::downsample], scaled_x[::downsample])
            u = self.x_amplitude[::downsample, ::downsample]
            v = self.y_amplitude[::downsample, ::downsample]
            a = self.intensity()[::downsample, ::downsample]
            ku = self.x_phase[::downsample, ::downsample]
            kv = self.y_phase[::downsample, ::downsample]
            ax1.quiver(X[mask], Y[mask], dx * downsample / 6 * u[mask], dx * downsample / 6 * v[mask], color='black', alpha=0.6)
        else:
            plt.title('Pupil amplitude')
        ax1.set_aspect('equal')
        ax1.set_xlim(-scaled_aperture / 1.9, scaled_aperture / 1.9)
        ax1.set_ylim(-scaled_aperture / 1.9, scaled_aperture / 1.9)
        if display_colorbar:
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
        precision: str = 'complex128',
        field: np.ndarray = None
    ):
        self._wavelength = wavelength
        self._numerical_aperture = numerical_aperture
        self._x = np.linspace(-field_diameter / 2, field_diameter / 2, shape[0])
        self._z = np.linspace(-field_depth / 2, field_depth / 2, shape[-1])
        if field is None:
            self._E = np.empty((*shape, 3), dtype=precision)  # X, Y, and Z field components along last axis
        else:
            self._E = field
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


def _worker(
    stop_event,
    work_event,
    barrier,
    workload,
    depths,
    n,
    z,
    precision,
    pupil_buf_name,
    field_buf_name,
    cztw_plan_args,
    kzxy,
    px_per_m,
    Gx,
    Gy,
    Gz        
):
    # Plan transform
    czt = cztw.plan(**cztw_plan_args)
    # Connect to shared memory
    pupil_buffer = sm.SharedMemory(name=pupil_buf_name, create=False)
    field_buffer = sm.SharedMemory(name=field_buf_name, create=False)
    pupil_array = np.ndarray((n, n, 2), dtype=precision, buffer=pupil_buffer.buf)
    field_array = np.ndarray((n, n, z, 3), dtype=precision, buffer=field_buffer.buf)
    print(os.getpid(), 'Planned transform. Connected to shared memory.', flush=True)
    while not stop_event.is_set():  # Main loop
        if work_event.is_set():
            for i, dz in zip(workload, depths):
                field_array[:, :, i, :] = _simulate_plane(pupil_array, dz, kzxy, px_per_m, czt, Gx, Gy, Gz)
            barrier.wait()
    pupil_buffer.close()
    field_buffer.close()


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
    
    return np.stack((E_Xx + E_Yx, E_Xy + E_Yy, E_Xz + E_Yz), axis=-1)  # Return E_X, E_Y, E_Z


class VectorialObjective(object):
    
    def __init__(
            self,
            wavelength: float,
            sample_index: float,
            focal_length: float,
            pupil_diameter: int,
            n: int,
            field_diameter: float,
            z: int = 1,
            field_depth: float = None,
            precision: str = 'complex128',
            multiprocessing: bool = False
        ):
        
        if precision in ['single', 'float', 'float32', 'complex64']:
            self._precision = 'complex64'
        else:
            self._precision = 'complex128'
        self._parallelized = bool(multiprocessing)
        if multiprocessing is True:
            self._n_processes = mp.cpu_count()
        elif type(multiprocessing) is int:
            self._n_processes = multiprocessing
        self._wavelength = wavelength
        self._index = sample_index
        self._focal_length = focal_length
        self._pupil_diameter = pupil_diameter        
        self._N = n  # Pupil and focal field planar sampling
        self._Z = z  # Depths to compute 
        self._field_diameter = field_diameter
        
        angle_of_convergence = np.arctan(pupil_diameter / (2 * focal_length))
        self._na = self._index * np.sin(angle_of_convergence)
        
        self._angle_of_convergence = np.arctan(self._pupil_diameter / (2 * self._focal_length))
        self._px_per_m = self._N / self._field_diameter
        self._m_per_px = self._field_diameter / self._N
        
        if self._Z > 1:
            if field_depth is None:  # Default to same sampling as lateral dimension
                self._field_depth = self._Z * self._m_per_px
            else:
                self._field_depth = field_depth
            zd = np.linspace(0, self._field_depth / 2, self._Z // 2 + 1)
            self._depths = np.concatenate((-zd[:0:-1], zd[:-1]))
        else:
            self._field_depth = 0
            self._depths = [0]
        
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
        self._ready = False  # True if _setup has run
        self._pupil_buf_shape = (self._N, self._N, 2)  # The size of the pupil buffer
        self._field_buf_shape = (self._N, self._N, self._Z, 3)  # The size of the field buffer
        self._czt = None  # Transform plan in non-parallel mode
        self._worker_pool = []  # Workers in parallel mode
        self._work_event = None
        self._abort_event = None
        self._worker_barrier = None
        self._shared_pupil_buf = None  # Shared memory for the pupil. Used to improve performance in parallel mode.  Allocated in setup stage.
        self._shared_field_buf = None  # Shared memory for the field. Used to improve performance in parallel mode. Allocated in setup stage.
        self._done = mp.Event()  # Set on cleanup
    
    def _setup(self):
        # The most time-consuming operations carried out the first time an Objective is used should be here
        if self._parallelized:
            # Set up sync system
            self._abort_event = mp.Event()
            self._work_event = mp.Event()
            self._worker_barrier = mp.Barrier(self._n_processes + 1)
            # self._worker_barrier = mp.Barrier(self._n_processes + 1)
            self._shared_pupil_buf = sm.SharedMemory(size=int(np.prod(self._pupil_buf_shape) * type_bytes[self._precision]), create=True)
            self._shared_field_buf = sm.SharedMemory(size=int(np.prod(self._field_buf_shape) * type_bytes[self._precision]), create=True)
            self._shared_pupil_array = np.ndarray(self._pupil_buf_shape, dtype=self._precision, buffer=self._shared_pupil_buf.buf)
            self._shared_field_array = np.ndarray(self._field_buf_shape, dtype=self._precision, buffer=self._shared_field_buf.buf)
            depths_per_worker = self._Z // self._n_processes
            workloads = [np.arange(depths_per_worker) + (depths_per_worker * i) for i in range(self._n_processes)]
            # Give extra jobs to the last worker
            while workloads[-1][-1] < self._Z - 1:
                workloads[-1] = np.append(workloads[-1], [workloads[-1][-1] + 1])
            for i, workload in enumerate(workloads):
                print('Giving worker', i, 'workload', (tuple(workload), tuple(self._depths[workload])))
                worker_args = {
                    'stop_event': self._abort_event,
                    'work_event': self._work_event,
                    'barrier': self._worker_barrier,
                    'workload': tuple(workload),
                    'depths': tuple(self._depths[workload]),
                    'n': self._N,
                    'z': self._Z,
                    'precision': self._precision,
                    'pupil_buf_name': self._shared_pupil_buf.name,
                    'field_buf_name': self._shared_field_buf.name,
                    'cztw_plan_args': dict(ndim=2, N=self._N, M=None, w0=-self._k_bandwidth, w1=self._k_bandwidth, precision=self._precision),
                    'kzxy': self._kzxy.copy(),
                    'px_per_m': self._px_per_m,
                    'Gx': self._Gx.copy(),
                    'Gy': self._Gy.copy(),
                    'Gz': self._Gz.copy()
                }
                self._worker_pool.append(mp.Process(target=_worker, kwargs=worker_args))
            for i, process in enumerate(self._worker_pool):
                print('Starting process', i)
                process.start()
        else:
            self._czt = cztw.plan(2, self._N, w0=-self._k_bandwidth, w1=self._k_bandwidth, precision=self._precision)
        self._ready = True

    def focus(self, pupil: VectorialPupil) -> VectorialFocalField:
        if not self._ready:
            self._setup()
        field = VectorialFocalField((self._N, self._N, self._Z), self._field_depth, self._field_diameter, self._wavelength, self._na)
        if self._parallelized and self._Z > 1:
            np.copyto(self._shared_pupil_array, pupil.xy)
            self._work_event.set()
            while self._worker_barrier.n_waiting < self._n_processes:
                pass
            self._work_event.clear()
            self._worker_barrier.wait()  # Releases workers
            np.copyto(field._E, self._shared_field_array)
        else:
            for i, dz in enumerate(self._depths):
                e = _simulate_plane(pupil.xy, dz, self._kzxy, self._px_per_m, self._czt, self._Gx, self._Gy, self._Gz)
                field._assign(i, e[:, :, 0], e[:, :, 1], e[:, :, 2])
        return field

    def __del__(self):
        if self._parallelized:
            self._abort_event.set()
            for process in self._worker_pool:
                process.join()
            self._shared_field_buf.unlink()
            self._shared_pupil_buf.unlink()
            
            
Objective = VectorialObjective  # Alias for VectorialObjective class
