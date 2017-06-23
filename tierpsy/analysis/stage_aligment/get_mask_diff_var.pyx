import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def get_mask_diff_var(np.ndarray[np.uint8_t, ndim=2] f, np.ndarray[np.uint8_t, ndim=2] g):
    #f, g are to one dim vector
    
    cdef int n_row = f.shape[0];
    cdef int n_col = f.shape[1];
    cdef int i, k
    cdef double var_diff;
    cdef double mean_x = 0;
    cdef double sum_x2 = 0;
    cdef double pix_diff = 0;
    cdef double tot_valid = 0;
    
    for i in range(n_row):
        for j in range(n_col):
            if ( f[i,j] != 0 ) and ( g[i,j] != 0 ):
                pix_diff = <double>f[i,j] - <double>g[i,j];
                mean_x += pix_diff;
                sum_x2 += pix_diff*pix_diff;
                tot_valid += 1;
    #print tot_pix
    if tot_valid > 0:
        mean_x /= tot_valid;
        var_diff = sum_x2/tot_valid - mean_x*mean_x;
        return var_diff;
    else:
        return -1;
    