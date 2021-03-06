import numpy as np 
from ctypes import CDLL, RTLD_GLOBAL, c_double, c_int, POINTER
import ctypes
import os , sys
import matplotlib.pyplot as plt 


libso_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])+'/'

# First get cudalc so 
dl = CDLL(libso_path+'cudalc.so', mode=RTLD_GLOBAL)


def make_nd_array(c_pointer, shape, dtype=np.float64, order='C', own_data=True):
    arr_size = np.prod(shape[:]) * np.dtype(dtype).itemsize 
    if sys.version_info.major >= 3:
        buf_from_mem = ctypes.pythonapi.PyMemoryView_FromMemory
        buf_from_mem.restype = ctypes.py_object
        buf_from_mem.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
        buffer = buf_from_mem(c_pointer, arr_size, 0x100)
    else:
        buf_from_mem = ctypes.pythonapi.PyBuffer_FromMemory
        buf_from_mem.restype = ctypes.py_object
        buffer = buf_from_mem(c_pointer, arr_size)
    arr = np.ndarray(tuple(shape[:]), dtype, buffer, order=order)
    if own_data and not arr.flags.owndata:
        return arr.copy()
    else:
        return arr

# then get kepler functions
def __get_kepler_functions():
    kepler = dl.kepler
    kepler.argtypes = [c_double, c_double, c_double]
    kepler.restype = c_double

    dkepler = dl.dkepler
    dkepler.argtypes = [c_double, c_double]
    dkepler.restype = c_double


    getEccentricAnomaly = dl.getEccentricAnomaly
    getEccentricAnomaly.argtypes = [c_double, c_double,c_int,c_double]
    getEccentricAnomaly.restype = c_double

    t_ecl_to_peri = dl.t_ecl_to_peri
    t_ecl_to_peri.argtypes = [c_double, c_double,c_double,c_double,c_double,c_double,c_double,c_double, c_int]
    t_ecl_to_peri.restype = c_double

    getTrueAnomaly = dl.getTrueAnomaly
    getTrueAnomaly.argtypes = [c_double, c_double,c_double,c_double,c_double,c_double,c_double,c_double,c_int,c_int,c_double]
    getTrueAnomaly.restype = c_double

    getProjectedPosition = dl.getProjectedPosition
    getProjectedPosition.argtypes = [c_double, c_double,c_double]
    getProjectedPosition.restype = c_double

    get_z = dl.get_z
    get_z.argtypes = [c_double, c_double,c_double, c_double,c_double]
    get_z.restype = c_double

    return kepler, dkepler, getEccentricAnomaly, t_ecl_to_peri,getTrueAnomaly, getProjectedPosition, get_z



kepler, dkepler, getEccentricAnomaly, t_ecl_to_peri, getTrueAnomaly, getProjectedPosition, get_z= __get_kepler_functions()

'''
    extern "C" {__device__ __host__ double kepler (double M, double E, double e);}
extern "C" {__device__ __host__ double dkepler (double E, double e);}

/************************************
*        Eccentric Anomaly          *
************************************/
extern "C" {__device__ __host__ double getEccentricAnomaly (double M, double e, int Accurate_Eccentric_Anomaly, double tol);}
extern "C" {__device__ __host__ double t_ecl_to_peri(double t_ecl, double e, double w, double incl, double radius_1, double p_sid, double t_ecl_tolerance, int Accurate_t_ecl);}
extern "C" {__device__ __host__ double getTrueAnomaly(double time, double  e, double w, double period, double t_zero, double incl, double radius_1, double t_ecl_tolerance, int Accurate_t_ecl,  int Accurate_Eccentric_Anomaly, double E_tol );}

/***************************************************
*        Calculate the projected seperaton         *
***************************************************/
extern "C" {__device__ __host__ double get_z(double nu, double e, double incl, double w, double radius_1) ;}
extern "C" {__device__ __host__ double get_z_(double nu, double * z);}

/***************************************************
*        Calculate the projected position          *
***************************************************/
extern "C" {__device__ __host__ double getProjectedPosition(double nu, double w, double incl);}

/*********************************************
*        Calculate the mass function         *
*********************************************/
extern "C" {__device__ __host__ double mass_function_1(double e, double P, double K1);}
extern "C" {__device__ __host__ double mass_function_1_(double M2, double * z);}
'''

'''
# Then get rv functions
def __get_rv_functions():
    _rv = dl.rv
    _rv.argtypes = [POINTER(c_double),
                 c_double, c_double,
                 c_double,c_double,c_double,
                 c_double,c_double,c_double,c_double,
                 c_double,c_double,c_double, c_int, c_double,c_double,
                 c_int, c_double, c_int, c_double,
                 POINTER(c_double), c_int, c_int]
    _rv.restype = None

    _rv_loglike = dl.rv_loglike

    _rv_loglike.argtypes = [POINTER(c_double),POINTER(c_double),POINTER(c_double), c_double,
                 c_double, c_double,
                 c_double,c_double,c_double,
                 c_double,c_double,c_double,c_double,
                 c_double,c_double,c_double, c_int, c_double,c_double,
                 c_int, c_double, c_int, c_double,
                 c_int]
    _rv_loglike.restype = c_double


    return _rv, _rv_loglike

_rv, _rv_loglike = __get_rv_functions()


def rv(time,
        t_zero = 0.0, period = 1.0,
        K1 = 10., fs=0., fc = 0.,
        V0 = 0., dV0 = 0., Vs = 0., Vc = 0.,
        radius_1 = 0.2, k=0.2, incl = 90.,
        ld_law_1=0,  ldc_1_1=0.8, ldc_1_2=0.8,
        Accurate_t_ecl=0, t_ecl_tolerance=1e-5,  Accurate_Eccentric_Anomaly=0, E_tol=1e-5):

    time = time.astype(np.float64)


    d_time =  time.ctypes.data_as(POINTER(c_double))
    RV = np.empty(time.shape[0], dtype = np.float64)
    d_RV =  RV.ctypes.data_as(POINTER(c_double))

    _rv (d_time, 
        t_zero, period, 
        K1, fs, fc, 
        V0, dV0, Vs, Vc, 
        radius_1, k, incl, ld_law_1, ldc_1_1, ldc_1_2,
        Accurate_t_ecl, t_ecl_tolerance,  Accurate_Eccentric_Anomaly, E_tol,
        d_RV,  0, time.shape[0])

    return make_nd_array(d_RV, (time.shape[0],))


def rv_loglike(time, RV, RV_err, jitter=0.1,
        t_zero = 0.0, period = 1.0,
        K1 = 10., fs=0., fc = 0.,
        V0 = 0., dV0 = 0., Vs = 0., Vc = 0.,
        radius_1 = 0.2, k=0.2, incl = 90.,
        ld_law_1=0,  ldc_1_1=0.8, ldc_1_2=0.8,
        Accurate_t_ecl=0, t_ecl_tolerance=1e-5,  Accurate_Eccentric_Anomaly=0, E_tol=1e-5):

    time = time.astype(np.float64)
    RV = RV.astype(np.float64)
    RV_err=RV_err.astype(np.float64)

    d_time =  time.ctypes.data_as(POINTER(c_double))
    d_RV=  RV.ctypes.data_as(POINTER(c_double))
    d_RV_err =  RV_err.ctypes.data_as(POINTER(c_double))


    return _rv_loglike (d_time, d_RV, d_RV_err, jitter,
        t_zero, period, 
        K1, fs, fc, 
        V0, dV0, Vs, Vc, 
        radius_1, k, incl, ld_law_1, ldc_1_1, ldc_1_2,
        Accurate_t_ecl, t_ecl_tolerance,  Accurate_Eccentric_Anomaly, E_tol,
        time.shape[0])

    #return RV


'''

# Now do LC
def __get_lc_functions():
    _lc = dl.lc
    _lc.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double),c_double,c_double,
                 c_double, c_double,
                 c_double,c_double,
                 c_double,c_double,
                 c_double, c_double,
                 c_double, c_double,
                 POINTER(c_double), c_double, c_int,
                 c_double,
                 c_int, c_double, c_double,c_double,
                 c_double, c_double,
                 c_int, c_double, c_int, c_double,
                 c_int, c_int,
                 c_int]
    _lc.restype = c_double


    _lc_gpu = dl.lc_gpu 
    _lc_gpu.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double),c_double,c_double,
                c_double, c_double,
                c_double,c_double,
                c_double,c_double,
                c_double, c_double,
                c_double, c_double,
                POINTER(c_double), c_double, c_int,
                c_double,
                c_int, c_double, c_double,c_double,
                c_double, c_double,
                c_int, c_double, c_int, c_double,
                c_int, c_int]
    _lc_gpu.restype = c_double


    return _lc, _lc_gpu


_lc, _lc_gpu = __get_lc_functions()

def lc(time, mag=np.zeros(1), mag_err=np.zeros(1), J=0., zp=0.0,
    t_zero = 0.0, period = 1.0,
    radius_1 = 0.2, k=0.2, 
    fs = 0.0, fc = 0.0, 
    q=0., albedo = 0.,
    alpha_doppler=0., K1 = 0.,
    spots = np.array([0.2, 0.0, 0.1,0.5]), omega_1=1., nspots=0,
    incl = 90.,
    ldc_law_1=0, ldc_1_1=0.8, ldc_1_2=0.8, gdc_1=0.3,
    SBR=0., light_3 = 0.,
    Accurate_t_ecl=0, t_ecl_tolerance=1e-5, Accurate_Eccentric_Anomaly=1, E_tol=1e-5,
    nthreads=1):

    if (mag[0]==0) or (mag_err[0]==0) : 
        LC = np.empty(time.shape[0], dtype = np.float64) # d_LC is the only thing that will be populated
        d_LC =  LC.ctypes.data_as(POINTER(c_double))
        d_LC_ERR =  LC.ctypes.data_as(POINTER(c_double))
        loglike_switch = 0
    else:
        d_LC = mag.ctypes.data_as(POINTER(c_double))
        d_LC_ERR = mag_err.ctypes.data_as(POINTER(c_double))
        loglike_switch = 1

    d_time =  time.ctypes.data_as(POINTER(c_double))
    d_spots = spots.ctypes.data_as(POINTER(c_double))

    loglike = _lc(d_time, d_LC, d_LC_ERR, J, zp,
        t_zero, period,
        radius_1, k,
        fs,fc,
        q,albedo,
        alpha_doppler, K1,
        d_spots, omega_1, nspots,
        incl,
        ldc_law_1, ldc_1_1, ldc_1_2, gdc_1,
        SBR, light_3,
        Accurate_t_ecl, t_ecl_tolerance, Accurate_Eccentric_Anomaly, E_tol,
        time.shape[0], nthreads,
        loglike_switch)

    if loglike_switch : return loglike
    else              : return make_nd_array(d_LC, (time.shape[0],))



def lc_gpu(time, mag=-99*np.ones(1), mag_err=-99*np.ones(1), J=0., zp=0.0,
    t_zero = 0.0, period = 1.0,
    radius_1 = 0.2, k=0.2, 
    fs = 0.0, fc = 0.0, 
    q=0., albedo = 0.,
    alpha_doppler=0., K1 = 0.,
    spots = np.array([0.2, 0.0, 0.1,0.5]), omega_1=1., nspots=0,
    incl = 90.,
    ldc_law_1=0, ldc_1_1=0.8, ldc_1_2=0.8, gdc_1=0.3,
    SBR=0., light_3 = 0.,
    Accurate_t_ecl=0, t_ecl_tolerance=1e-5, Accurate_Eccentric_Anomaly=1, E_tol=1e-5,
    threads_per_block = 512):

    
    if (mag[0]==-99) or (mag_err[0]==-99) : 
        LC = np.empty(time.shape[0], dtype = np.float64) # d_LC is the only thing that will be populated
        d_LC =  LC.ctypes.data_as(POINTER(c_double))
        d_LC_ERR =  LC.ctypes.data_as(POINTER(c_double))
        loglike_switch = 0
    else:
        d_LC = mag.ctypes.data_as(POINTER(c_double))
        d_LC_ERR = mag_err.ctypes.data_as(POINTER(c_double))
        loglike_switch = 1


    d_time =  time.ctypes.data_as(POINTER(c_double))
    d_spots = spots.ctypes.data_as(POINTER(c_double))

    loglike = _lc_gpu(d_time, d_LC, d_LC_ERR, J, zp,
        t_zero, period,
        radius_1, k,
        fs,fc,
        q,albedo,
        alpha_doppler, K1,
        d_spots, omega_1, nspots,
        incl,
        ldc_law_1, ldc_1_1, ldc_1_2, gdc_1,
        SBR, light_3,
        Accurate_t_ecl, t_ecl_tolerance, Accurate_Eccentric_Anomaly, E_tol,
        time.shape[0], threads_per_block)

    return loglike





