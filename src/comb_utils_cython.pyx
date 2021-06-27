# cython: language_level=3
import numpy as np
cimport cython
from cython.parallel import prange
import pyfftw
# from libc.complex cimport exp as cexp
# from libc.complex cimport abs as cabs

cdef extern from "complex.h":
    double complex cexp(double complex z) nogil
    # double complex abs(double complex y) #nogil

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef double complex[:,::1] split_step_cython(unsigned int Nsim, double complex dt,
    double complex[::1] dwext,double complex dwi,double complex dwtot,
    double complex[::1] Dint_arr,double complex[::1] detuning,
    double complex[::1] F_arr, unsigned int Nmodes,
    double complex[::1] noise_norm, unsigned int divbysnapshot,
    unsigned int Nsnapshots,pbar,ifft_object,fft_object,
    double complex[::1] ifft_arr, double complex[::1] fft_arr,
    const int Nthreads): # ifft_object,fft_object, double complex[::1] ifft_arr, double complex[::1] fft_arr, 
    
    cdef unsigned int Dlen = dwext.shape[0]
    cdef double complex[::1] Dispersion = np.zeros((Dlen),np.complex128)
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int sol_track = 0
    sol = np.zeros((Nsnapshots,Dlen),np.complex128) # cdef double complex[:,::1] 
    
    for i in range(Nsim):
        for j in prange(Dlen,nogil=True,num_threads=Nthreads): # j in range(Dlen):#
            Dispersion[j] = cexp(-(dt/2)*((dwext[j]+dwi)/dwtot+1j*(Dint_arr[j] + detuning[i])*2/dwtot ))
            ifft_arr[j] *= Dispersion[j]
        ifft_object()
        for j in prange(Dlen,nogil=True,num_threads=Nthreads): # j in range(Dlen):#
            fft_arr[j] = cexp(dt *(1j *  abs(fft_arr[j]) ** 2 + (F_arr[j]/fft_arr[j]))) * fft_arr[j]
        fft_object()
        for j in prange(Dlen,nogil=True,num_threads=Nthreads): # j in range(Dlen):#
            ifft_arr[j] *= Dispersion[j]
            ifft_arr[j] += noise_norm[j]
        if (not((i+1)%divbysnapshot)) and sol_track<Nsnapshots:
            sol[sol_track,:] = ifft_arr
            sol_track += 1
            pbar.update(1)
    return sol