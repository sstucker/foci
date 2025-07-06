# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:23:46 2025

@author: sstucker
"""

import os
import pickle
import numpy as np
import pyfftw as fftw


# Global package units
M = 1.
MM = M / 1000.
UM = MM / 1000.
NM = UM / 1000.


try:
    fftw_wisdom_path = os.path.join(os.path.dirname(__file__), 'fftw_wisdom')
except NameError:
    fftw_wisdom_path = os.path.join(os.getcwd(), 'fftw_wisdom')


def import_fftw_wisdom():
    wisdom = []
    if os.path.exists(fftw_wisdom_path):
        with open(fftw_wisdom_path, 'rb') as f:
            fftw.import_wisdom(pickle.load(f))


def export_fftw_wisdom():
    wisdom = fftw.export_wisdom()
    with open(fftw_wisdom_path, 'wb') as f:
        pickle.dump(fftw.export_wisdom(), f)
            