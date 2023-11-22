# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:25:33 2023

@author: sstucker

"Fastest Chirp-Z Transform in the West"

Chirp Z-transform (Bluestein's algorithm) implemented with FFTW.
                        
CPU-bound version of the implementation in (2).
                        
[1] Bluestein, Leo I. "A linear filtering approach to the computation of the discrete Fourier transform," Northeast Electronics Research and Engineering Meeting Record 10, 218-219 (1968).
[2] Vishniakou, I. & Seelig, J. D. "Differentiable optimization of the Debye-Wolf integral for light shaping and adaptive optics in two-photon microscopy". Opt. Express, OE 31, 9526–9542 (2023).

"""

import numpy as np
from numpy import pi as PI
import pyfftw as fftw
import os


wisdom_path = os.path.join(os.getcwd(), 'wisdom')  # todo save this in module directory


def next_power_of_2(a) -> int:
    return 2**(a - 1).bit_length()
    

class CZTW(object):
    
    def __init__(self, ndim, N, M, w0=-1/2, w1=1/2, precision='double', planner_effort='FFTW_PATIENT'):
        if precision in ['double, complex128', '128']:
            self._complex_dtype = 'complex128'
            self._real_dtype = 'float64'
        else:
            self._complex_dtype = 'complex64'
            self._real_dtype = 'float32'
       
        self._ndim = ndim
        if self._ndim not in (1, 2):
            raise ValueError('Only 1 and 2 dimensional CZTs are supported.')
       
        L = next_power_of_2(N + M)
        self._L = L
        self._N = N
        self._M = M 

        self._input_shape = tuple([N for _ in range(ndim)])
        self._output_shape = tuple([M for _ in range(ndim)])
       
        n = np.arange(0, N)
        k = np.arange(0, M)
        r = np.arange(L - N, L)
        
        A = 1.0 * np.exp(2j * PI * w0)
        dw = -2j * PI * (w1 - w0) / M
        W = 1.0 * np.exp(dw)
        
        z = A * W**-k
        self._y = A**-n * W**(n**2/2)
        
        v_n = np.zeros([self._L], dtype=self._complex_dtype)
        v_n[:M] = W**(-k**2 / 2)
        v_n[L - N:L + 1] = W**(-(L - r)**2 / 2)
        
        self._V = np.fft.fft(v_n)
        self._g = W**(k**2/2)
        
        wisdom = []
        if os.path.exists(wisdom_path):
            with open(wisdom_path, 'rb') as f:
                for line in f:
                    wisdom.append(line)
            fftw.import_wisdom(wisdom)
        if ndim == 2:
            # if N != M, the FFT changes size across the 2nd axis and 4 transforms/arrays must be planned.
            self._transform_shape = (M, L)
            self._x1 = fftw.empty_aligned((N, L), dtype=self._complex_dtype)
            self._x2 = fftw.empty_aligned((M, L), dtype=self._complex_dtype)
            self._X1 = fftw.empty_aligned((N, L), dtype=self._complex_dtype)
            self._X2 = fftw.empty_aligned((M, L), dtype=self._complex_dtype)
            self._fft_fwd_i = fftw.builders.fft2(self._x1, axes=(True, False), overwrite_input=True, avoid_copy=True, threads=1, planner_effort=planner_effort, auto_align_input=True)
            self._fft_fwd_j = fftw.builders.fft2(self._x2, axes=(True, False), overwrite_input=True, avoid_copy=True, threads=1, planner_effort=planner_effort, auto_align_input=True)
            self._fft_bwd_i = fftw.builders.ifft2(self._X1, axes=(True, False), overwrite_input=True, avoid_copy=True, threads=1, planner_effort=planner_effort, auto_align_input=True)
            self._fft_bwd_j = fftw.builders.ifft2(self._X2, axes=(True, False), overwrite_input=True, avoid_copy=True, threads=1, planner_effort=planner_effort, auto_align_input=True)
            self._fft_fwd_i(self._x1)
            self._fft_fwd_j(self._x2)
            self._fft_bwd_i(self._X1)
            self._fft_bwd_j(self._X2)
            self.czt = self._czt2
        if ndim == 1:
            self._transform_shape = (L) 
            self._x = fftw.empty_aligned(self._transform_shape, dtype=self._complex_dtype)
            self._X = fftw.empty_aligned(self._transform_shape, dtype=self._complex_dtype)
            self._fft_fwd = fftw.builders.fft(self._x, overwrite_input=True, avoid_copy=True, threads=1, planner_effort=planner_effort, auto_align_input=True)
            self._fft_bwd = fftw.builders.ifft(self._X, overwrite_input=True, avoid_copy=True, threads=1, planner_effort=planner_effort, auto_align_input=True)
            self._fft_fwd(self._x)
            self._fft_Bwd(self._X)
            self.czt = self._czt1
        wisdom = fftw.export_wisdom()
        with open(wisdom_path, 'wb') as f:
            for line in wisdom:
                f.write(line)
            
    def __call__(self, x: np.ndarray):
        # if x.shape != self._input_shape:
        #     raise ValueError('Invalid input shape {} for CZTW plan with input shape {}'.format(x.shape, self._input_shape))
        return self.czt(x)
        
    def _czt1(self, x: np.ndarray):
        self._x[:] = 0
        self._X[:] = 0
        self._x[:self._N] = x * self._y
        self._X = self._fft_fwd(self._x)
        self._x = self._fft_bwd(self._V * self._X)
        return self._x[:self._M] * self._g
    
    def _czt2(self, x: np.ndarray):
        self._x1[:] = 0
        self._x2[:] = 0
        self._X1[:] = 0
        self._X2[:] = 0
        self._x1[:, :self._N] = x * self._y
        self._X1 = self._fft_fwd_i(self._x1)
        self._x1 = self._fft_bwd_i(self._V * self._X1)
        self._x2[:, :self._N] = np.transpose(self._x1[:, :self._M] * self._g) * self._y
        self._X2 = self._fft_fwd_j(self._x2)
        self._x2 = self._fft_bwd_j(self._V * self._X2)
        return np.transpose(self._x2[:, :self._M] * self._g)
    

def plan(ndim, N, M=None, w0=-1/2, w1=1/2, precision='double') -> CZTW:
    if M is None:
        M = N
    return CZTW(ndim, N, M, w0=w0, w1=w1, precision=precision)
