# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 17:54:42 2015

@author: ajaver
"""
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t

cdef inline double rationR(double a, double b): a/b if a>=b else b; #f[i]/g[j] if f[i] >= g[j] else g[j]/f[i]

@cython.boundscheck(False)
def calculate_ratio(np.ndarray[DTYPE_t, ndim=1] f, np.ndarray[DTYPE_t, ndim=1] g):
    #f, g are to one dim vector
    
    cdef int nf = f.size;
    cdef int ng = g.size;
    cdef int i, j
    cdef np.ndarray[DTYPE_t, ndim=2] mp = np.zeros([nf, ng], dtype=f.dtype);
    
    for i in range(nf):
        for j in range(ng):
            mp[i,j] = rationR(f[i],g[j])
            
    return mp