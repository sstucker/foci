# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:33:03 2023

@author: sstucker
"""

import numpy as np
from numpy import pi as PI
import pyfftw as fftw


def wpm(λ, u0, x, dz, z_steps, n=None, precise=True, verbose=False) -> np.ndarray:
    """
    Schmidt wave propagation method.
    [1] S. Schmidt, T. Tiess, S. Schröter, R. Hambach, M. Jäger, H. Bartelt, A. Tünnermann, and H. Gross, “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express 24(26), 30188–30200 (2016).
    Parameters
    ----------
    λ : float
        Wavelength.
    u0 : np.ndarray (complex)
        Incident field.
    x : np.ndarray (float)
        Spatial coordinate.
    dz : float
        Propagation step size.
    z_steps : int
        Number of propagation steps.
    n : np.ndarray (float), optional
        2D index of refraction structure. The default is None.
    precise : bool, optional
        Whether to use 128 or 64 bitness. The default is True.
    verbose : bool, optional
        Print progress bar. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    if precise:
        dtype_s = 'complex128'
        dtype_np = np.complex128
    else:
        dtype_s = 'complex64'
        dtype_np = np.complex64
    u0 = u0.astype(dtype_np)
    x = x.astype(dtype_np)    
    u = np.empty((len(x), z_steps + 1), dtype=dtype_np)
    u[:, 0] = u0  # Incident field is first slice of u
    if n is None: # default is free space propagation
        n = np.ones(u.shape, dtype=dtype_np) * (1 + 0j)
    elif n.shape[0] != x.shape or n.shape[1] != z_steps:
        return IndexError("'n' must be same transverse size as incident field with second dimension equal to 'z_steps'.")
    k = 2 * PI / λ
    dx = np.abs(x[1] - x[0])
    kx = 2 * np.pi / (x.size * dx) * (range(-int(x.size / 2), int(x.size / 2)))
    kx2 = kx**2
    if verbose:
        print('-- 2D Wave Propagation Method ----------')
        print('Precision: ' + dtype_s)
        print('Incident field size: {} ({:.4} GB)'.format(len(u0), (len(u0) * u0.itemsize) /  2**30))
        print('Resultant field size: {}x{} ({:.4} GB)'.format(len(u0), z_steps, (len(u0) * z_steps * u0.itemsize) /  2**30))
    if λ / dx < 32:
        print('Poor transverse discretization of {:.3} λ/px!'.format(λ / dx))
    if λ / dz < 32:
        print('Poor axial discretization of {:.3} λ/px!'.format(λ / dz))

    # Prepare FFTW
    u_prev = fftw.empty_aligned(x.size, dtype=dtype_s)
    E = fftw.empty_aligned(x.size, dtype=dtype_s)
    HxE = fftw.empty_aligned(x.size, dtype=dtype_s)
    e = fftw.empty_aligned(x.size, dtype=dtype_s)
    
    if z_steps > 1024:
        flags = ('FFTW_PATIENT',)
    else:
        flags = ('FFTW_ESTIMATE',)
    
    fft = fftw.FFTW(u_prev, E, direction='FFTW_FORWARD', flags=flags)
    ifft = fftw.FFTW(HxE, e, direction='FFTW_BACKWARD', flags=flags)
    
    frame_times = []
    
    start = time.time()
    last_updated = start
    for z in range(1, z_steps + 1):
        # Compute each index component separately and add
        start_frame = time.time()
        for m, n_m in enumerate(np.unique(n[:, z - 1])):
            Imz = (n[:, z - 1] == n_m)
            # Plane wave decomposition from [1]
            u_prev[:] = u[:, z - 1] # Copy into FFTW buffer
            fft() # E = fft(u[:, z - 1])
            H = np.exp(1j * dz * np.emath.sqrt(n_m**2 * k**2 - kx2))
            HxE[:] = fftshift(fftshift(E) * H)  # Copy into FFTW buffer
            ifft()  # e = ifft(HxE)
            u[:, z] += e * Imz  # Copy out of FFTW buffer
        if verbose:
            frame_times.append(time.time() - start_frame)
            if round(time.time() - last_updated) > 15.0:  # Update every 15 seconds
                last_updated = time.time()
                print('{:.2f} s elapsed. {}/{} steps '.format(time.time() - start, z, z_steps) + progress_bar(z / z_steps))
    if verbose:
        print('Finished!')
        print('Average propagation step took {:.3} s'.format(np.mean(frame_times)))
        print('Total time: {:.4} s'.format(time.time() - start))
        print('----------------------------------------')
    # return u[:, 1:]
    return u
