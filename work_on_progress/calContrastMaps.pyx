# -*- coding: utf-8 -*-
# cython: profile=True
"""
Created on Fri Feb 13 17:54:42 2015

@author: ajaver
"""


import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport round as c_round;
from libc.math cimport sqrt, floor


@cython.profile(False)
cdef inline int absDiff(int a, int b): 
    return a-b if a>b else b-a;

cdef inline int calcR(int a, int b):
    cdef double R;
    R = <double>(a*a + b*b);
    R = (sqrt(R));
    return <int>R

@cython.profile(False)
cdef inline int calcRScaling(int a, int b, double scaling):
    return <int>(scaling*sqrt(<double>(a*a + b*b)));

@cython.profile(False)
cdef inline int absDiffScaling(int a, int b, double scaling): 
    return <int>(scaling*absDiff(a,b));

@cython.profile(False)
cdef inline int sumScaling(int a, int b, double scaling): 
    return <int>(scaling*(a+b));


def calContrastMaps(np.ndarray[np.int64_t, ndim=2] pix_dat, int map_R_range, int map_pos_range, int map_neg_range):
    cdef np.ndarray[np.int_t, ndim=2] Ipos = np.zeros([map_R_range, map_pos_range], dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=2] Ineg = np.zeros([map_R_range, map_neg_range], dtype=np.int)
    
    cdef int i1, i2, ipos, ineg;
    cdef int n_pix = pix_dat.shape[1];
    cdef int R, delX, delY;
    
    for i1 in range(n_pix-1):
        for i2 in range(i1+1, n_pix):
            ipos = pix_dat[2,i1] + pix_dat[2,i2];
            ineg = absDiff(pix_dat[2,i1] , pix_dat[2,i2])
            
            delX = pix_dat[0,i1]-pix_dat[0,i2];
            delY = pix_dat[1,i1]-pix_dat[1,i2];
            
            R = calcR(delX, delY);
            if R>=map_R_range:
                R = map_R_range-1;
            
            Ipos[R, ipos] += 1;
            Ineg[R, ineg] += 1;
            
    return (Ipos, Ineg);


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def calContrastMapsBinned(np.ndarray[np.int64_t, ndim=2] pix_dat, dict bins_size, dict max_values):
    cdef np.ndarray[np.int_t, ndim=2] Ipos = np.zeros([bins_size['R'], bins_size['pos']], dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=2] Ineg = np.zeros([bins_size['R'], bins_size['neg']], dtype=np.int)
    
    cdef int i1, i2, ipos, ineg;
    cdef int n_pix = pix_dat.shape[1];
    cdef int R, delX, delY;
    
    cdef double pos_scaling = bins_size['pos']/max_values['pos'];
    cdef double neg_scaling = bins_size['neg']/max_values['neg'];
    cdef double R_scaling = bins_size['R']/max_values['R'];
    cdef int R_bins = <int>bins_size['R']
    
    
    for i1 in range(n_pix-1):
        for i2 in range(i1+1, n_pix):
            
            ipos = sumScaling(pix_dat[2,i1], pix_dat[2,i2], pos_scaling);
            ineg = absDiffScaling(pix_dat[2,i1] , pix_dat[2,i2], neg_scaling);
            
            delX = pix_dat[0,i1]-pix_dat[0,i2];
            delY = pix_dat[1,i1]-pix_dat[1,i2];
            
            R = calcRScaling(delX, delY, R_scaling);
            if R>=R_bins:
                R = R_bins-1;
            
            Ipos[R, ipos] += 1;
            Ineg[R, ineg] += 1;
            
    return (Ipos, Ineg);