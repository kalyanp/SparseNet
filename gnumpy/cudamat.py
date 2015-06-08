import os, pdb, platform, time, warnings
import ctypes as ct
import numpy as np

MAX_ONES = 1024*256

class CudaDeviceProp(ct.Structure):
    _fields_ = [
        ("name",ct.c_char*256),
        ("totalGlobalMem",ct.c_size_t),
        ("sharedMemPerBlock",ct.c_size_t),
        ("regsPerBlock",ct.c_int),
        ("warpSize",ct.c_int),
        ("memPitch",ct.c_size_t),
        ("maxThreadsPerBlock",ct.c_int),
        ("maxThreadsDim[3]",ct.c_int*3),
        ("maxGridSize[3]",ct.c_int*3),
        ("clockRate",ct.c_int),
        ("totalConstMem",ct.c_size_t),
        ("major",ct.c_int),
        ("minor",ct.c_int),
        ("textureAlignment",ct.c_size_t),
        ("texturePitchAlignment",ct.c_size_t),
        ("deviceOverlap",ct.c_int),
        ("multiProcessorCount",ct.c_int),
        ("kernelExecTimeoutEnabled",ct.c_int),
        ("integrated",ct.c_int),
        ("canMapHostMemory",ct.c_int),
        ("computeMode",ct.c_int),
        ("maxTexture1D",ct.c_int),
        ("maxTexture1DMipmap",ct.c_int),
        ("maxTexture1DLinear",ct.c_int),
        ("maxTexture2D[2]",ct.c_int*2),
        ("maxTexture2DMipmap[2]",ct.c_int*2),
        ("maxTexture2DLinear[3]",ct.c_int*3),
        ("maxTexture2DGather[2]",ct.c_int*2),
        ("maxTexture3D[3]",ct.c_int*3),
        ("maxTextureCubemap",ct.c_int),
        ("maxTexture1DLayered[2]",ct.c_int*2),
        ("maxTexture2DLayered[3]",ct.c_int*3),
        ("maxTextureCubemapLayered[2]",ct.c_int*2),
        ("maxSurface1D",ct.c_int),
        ("maxSurface2D[2]",ct.c_int*2),
        ("maxSurface3D[3]",ct.c_int*3),
        ("maxSurface1DLayered[2]",ct.c_int*2),
        ("maxSurface2DLayered[3]",ct.c_int*3),
        ("maxSurfaceCubemap",ct.c_int),
        ("maxSurfaceCubemapLayered[2]",ct.c_int*2),
        ("surfaceAlignment",ct.c_size_t),
        ("concurrentKernels",ct.c_int),
        ("ECCEnabled",ct.c_int),
        ("pciBusID",ct.c_int),
        ("pciDeviceID",ct.c_int),
        ("pciDomainID",ct.c_int),
        ("tccDriver",ct.c_int),
        ("asyncEngineCount",ct.c_int),
        ("unifiedAddressing",ct.c_int),
        ("memoryClockRate",ct.c_int),
        ("memoryBusWidth",ct.c_int),
        ("l2CacheSize",ct.c_int),
        ("maxThreadsPerMultiProcessor",ct.c_int)]


dllext = 'dll' if platform.system() == 'Windows' else 'so'
dllpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'libcudamat.%s' % dllext)

_cudamat = ct.cdll.LoadLibrary(dllpath)

_cudamat.get_last_cuda_error.restype = ct.c_char_p
_cudamat.cublas_init.restype = ct.c_int
_cudamat.cublas_shutdown.restype = ct.c_int
_cudamat.cuda_set_device.restype = ct.c_int
_cudamat.cuda_memory_available.restype = ct.c_size_t
_cudamat.cuda_memory_total.restype = ct.c_size_t
_cudamat.cuda_get_device.restype = ct.c_int
_cudamat.cuda_get_device_count.restype = ct.c_int
_cudamat.cuda_get_device_prop.restype = ct.c_int
_cudamat.init_random.restype = ct.c_int

_cudamat.init_empty.restype = ct.c_int
_cudamat.reshape.restype = ct.c_int
_cudamat.copy_to_host.restype = ct.c_int
_cudamat.allocate_device_memory = ct.c_int
_cudamat.copy_to_device.restype = ct.c_int
_cudamat.copy_on_device.restype = ct.c_int
_cudamat.free_device_memory.restype = ct.c_int

_cudamat.get_slice.restype = ct.c_int
_cudamat.get_row_slice.restype = ct.c_int
_cudamat.set_row_slice.restype = ct.c_int
_cudamat.copy_transpose.restype = ct.c_int
_cudamat.get_vector_slice.restype = ct.c_int
_cudamat.fill_with_rand.restype = ct.c_int
_cudamat.fill_with_randn.restype = ct.c_int

_cudamat.add_col_vec.restype = ct.c_int
_cudamat.sub_col_vec.restype = ct.c_int
_cudamat.add_col_mult.restype = ct.c_int
_cudamat.add_row_vec.restype = ct.c_int
_cudamat.sub_row_vec.restype = ct.c_int
_cudamat.mult_by_col_vec.restype = ct.c_int
_cudamat.mult_by_row_vec.restype = ct.c_int
_cudamat.clip_norm.restype = ct.c_int

_cudamat.less_than.restype = ct.c_int
_cudamat.less_than_scalar.restype = ct.c_int
_cudamat.greater_than.restype = ct.c_int
_cudamat.greater_than_scalar.restype = ct.c_int
_cudamat.maximum.restype = ct.c_int
_cudamat.maximum_scalar.restype = ct.c_int
_cudamat.min_by_axis.restype = ct.c_int
_cudamat.max_by_axis.restype = ct.c_int
_cudamat.sign.restype = ct.c_int
_cudamat.apply_relu.restype = ct.c_int
_cudamat.apply_sigmoid.restype = ct.c_int
_cudamat.apply_sigmoid_deriv.restype = ct.c_int
_cudamat.apply_tanh.restype = ct.c_int
_cudamat.apply_tanh_deriv.restype = ct.c_int
_cudamat.apply_abs.restype = ct.c_int
_cudamat.apply_log_1_plus_exp.restype = ct.c_int
_cudamat.apply_log.restype = ct.c_int
_cudamat.apply_exp.restype = ct.c_int
_cudamat.apply_sqrt.restype = ct.c_int
_cudamat.apply_pow.restype = ct.c_int
_cudamat.apply_pow_matrix.restype = ct.c_int
_cudamat.reciprocal.restype = ct.c_int
_cudamat.square.restype = ct.c_int
_cudamat.dropout.restype = ct.c_int

_cudamat.add_elementwise.restype = ct.c_int
_cudamat.add_transpose.restype = ct.c_int
_cudamat.subtract_elementwise.restype = ct.c_int
_cudamat.subtract_transpose.restype = ct.c_int
_cudamat.divide_elementwise.restype = ct.c_int
_cudamat.mult_elementwise.restype = ct.c_int
_cudamat.assign_array.restype = ct.c_int
_cudamat.assign_scalar.restype = ct.c_int
_cudamat.mult_by_scalar.restype = ct.c_int
_cudamat.divide_by_scalar.restype = ct.c_int
_cudamat.add_scalar.restype = ct.c_int

_cudamat.euclid_norm.restype = ct.c_double
_cudamat.selectRows.restype = ct.c_int
_cudamat.setSelectedRows.restype = ct.c_int
_cudamat.vdot.restype = ct.c_double
_cudamat.dot.restype = ct.c_int
_cudamat.diff_cols.restype = ct.c_int
_cudamat.diff_rows.restype = ct.c_int

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc

class CUDAMatException(Exception):
    pass

def get_last_cuda_error():
    return str(_cudamat.get_last_cuda_error())

def generate_exception(err_code):
    """
    Return a CUDAMatException object based on the error code err_code.
    """

    if err_code == -1:
        return CUDAMatException("Incompatible matrix dimensions.")
    elif err_code == -2:
        return CUDAMatException("CUBLAS error.")
    elif err_code == -3:
        return CUDAMatException("CUDA error: " + get_last_cuda_error())
    elif err_code == -4:
        return CUDAMatException("Operation not supported on views.")
    elif err_code == -5:
        return CUDAMatException("Operation not supported on transposed matrices.")
    elif err_code == -6:
        return CUDAMatException("")
    elif err_code == -7:
        return CUDAMatException("Incompatible transposedness.")
    elif err_code == -8:
        return CUDAMatException("Matrix is not in device memory.")
    elif err_code == -9:
        return CUDAMatException("Operation not supported.")
    elif err_code == -10:
        return CUDAMatException("Mismatched data types.")
    elif err_code == -11:
        return CUDAMatException("Operation not supported on this data type.")
        

global supported_dtypes
supported_dtypes = ['float32','float64','uint8','uint32']

def dtype_np2cm(dtype):
    if dtype == 'float32': return 0
    if dtype == 'float64': return 1
    if dtype == 'uint8':   return 2
    if dtype == 'uint32':  return 3
    raise Exception("cudamat error: unsupported dtype '%s'" % str(dtype))

def dtype_cm2np(dtype):
    if dtype == 0: return 'float32'
    if dtype == 1: return 'float64'
    if dtype == 2: return 'uint8'
    if dtype == 3: return 'uint32'
    raise Exception("cudamat error: unrecognized dtype '%s'" % str(dtype))

class cudamat(ct.Structure):
    _fields_ = [('data_host', ct.c_void_p),
                ('data_device', ct.c_void_p),
                ('on_device', ct.c_int),
                ('on_host', ct.c_int),
                ('size', ct.c_int * 2),
                ('is_trans', ct.c_int),
                ('owns_data', ct.c_int),
                ('dtype', ct.c_int)]

class rnd_struct(ct.Structure):
    _fields_ = [('dev_rnd_mults', ct.POINTER(ct.c_uint)), 
                ('dev_rnd_words', ct.POINTER(ct.c_longlong))]

class TransposedCUDAMatrix(object):
    def __init__(self, mat):
        self.mat = cudamat()
        ct.memmove(ct.pointer(self.mat), ct.pointer(mat), ct.sizeof(self.mat))
        self.mat.size[0] = mat.size[0] ############
        self.mat.size[1] = mat.size[1] ############             
        self.mat.is_trans = 1
        self.p_mat = ct.pointer(self.mat)

class CUDAMatrix(object):
    """
    A CUDAMatrix object represents a matrix of single precision floating point
    numbers on a GPU.
    """

    def __init__(self, array, copy_to_device = True):
        """
        Initializes a new matrix object in one of two ways. If array is a numpy
        ndarray, memory for a matrix with the same dimensions is allocated on
        the GPU. If the copy_to_device flag is set to True, the GPU matrix is
        initialized with the given ndarray. If array is not an ndarray, it must
        be a cudamat structure (typically the user will never use this way of
        calling __init__).
        """

        if type(array) == np.ndarray:
            # Convert array to float32 in FORTRAN order
            array = reformat(array)

            # Initialize as a ndarray-tied matrix.
            self.mat = cudamat()
            self.size = self.mat.size
            self.p_mat = ct.pointer(self.mat)
            self.numpy_array = array

            _cudamat.init_from_array(self.p_mat, array.ctypes.data_as(ct.c_void_p), ct.c_int(array.shape[0]), ct.c_int(array.shape[1]), ct.c_int(dtype_np2cm(array.dtype)))
            if copy_to_device:
                err_code = _cudamat.copy_to_device(self.p_mat)
                if err_code:
                    raise generate_exception(err_code)

        else:
            # Initialize based on existing cudamat structure.
            mat = array
            self.mat = mat
            self.p_mat = ct.pointer(self.mat)

        self.T = TransposedCUDAMatrix(self.mat)

        # Keep a reference to free device memory in case of a crash.
        self.__free_device_memory = _cudamat.free_device_memory


    def __del__(self):
        try:
            if 'p_mat' in self.__dict__:
                err_code = self.__free_device_memory(self.p_mat)
                if err_code:
                    raise generate_exception(err_code)
        except AttributeError:
            pass

    @staticmethod
    def init_random(seed = 0):
        """
        Initialize and seed the random number generator.
        """

        NUM_RND_STREAMS = 96*128
        CUDAMatrix.rndInitialized = 1
        CUDAMatrix.rnd_state = rnd_struct()
        CUDAMatrix.rnd_state_p = ct.pointer(CUDAMatrix.rnd_state)

        cudamat_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rnd_multipliers_32bit.txt')

        err_code = _cudamat.init_random(CUDAMatrix.rnd_state_p, ct.c_int(seed), cudamat_path)
        if err_code:
            raise generate_exception(err_code)

    @property
    def shape(self):
        return (self.mat.size[0], self.mat.size[1])

    def dtype(self):
        return dtype_cm2np(self.mat.dtype)

    def reshape(self, shape):
        """
        Reshapes self to have the given shape. The number of elements cannot
        change as this only changes how the contents are interpreted.
        """

        m = ct.c_uint(shape[0])
        n = ct.c_uint(shape[1])

        err_code = _cudamat.reshape(self.p_mat, m, n)
        if err_code:
            raise generate_exception(err_code)

        return self

    def asarray(self):
        """
        Copies the matrix to an ndarray on the CPU and returns it.
        """

        self.copy_to_host()

        return self.numpy_array

    def copy_to_device(self):
        """
        Copy the matrix to the GPU.
        """

        err_code = _cudamat.copy_to_device(self.p_mat)
        if err_code:
            raise generate_exception(err_code)

    def copy_to_host(self):
        """
        Copy the matrix to the CPU.
        """

        if not self.mat.on_host:
            # allocate host storage if necessary
            m = self.mat.size[0]
            n = self.mat.size[1]

            self.numpy_array = np.empty((m, n), dtype=dtype_cm2np(self.mat.dtype), order = 'F')
            self.mat.data_host = self.numpy_array.ctypes.data_as(ct.c_void_p)

            self.mat.on_host = 1

        err_code = _cudamat.copy_to_host(self.p_mat)
        if err_code:
            raise generate_exception(err_code)

    def assign(self, val):
        """Assign val to self, where val can be a scalar or a CUDAMatrix
        with the same dimensions as self. """

        err_code = 0
        if isinstance(val, CUDAMatrix):
            if val.mat.dtype == self.mat.dtype:
                err_code = _cudamat.copy_on_device(val.p_mat, self.p_mat)
            else:
                err_code = _cudamat.assign_array(val.p_mat, self.p_mat)
        elif isinstance(val, (int, float)):
            err_code = _cudamat.assign_scalar(self.p_mat, ct.c_double(val))
        else:
            raise ValueError, "Assigned value must be of type CUDAMatrix, int, or float."
            
        if err_code:
            raise generate_exception(err_code)

        return self

    def free_device_memory(self):
        """
        Free memory used up by the matrix on the GPU.
        """

        err_code = _cudamat.free_device_memory(self.p_mat)
        if err_code:
            raise generate_exception(err_code)

    def set_trans(self, is_trans):
        """
        Set the transposedness flag to is_trans.
        """

        _cudamat.set_transpose(self.p_mat, ct.c_int(1 * is_trans))

    def slice(self, first_col, last_col):
        mat = cudamat()

        if self.mat.size[0] == 1 or self.mat.size[1] == 1:
            err_code = _cudamat.get_vector_slice(self.p_mat, ct.pointer(mat), ct.c_int(first_col), ct.c_int(last_col))
        else:
            err_code = _cudamat.get_slice(self.p_mat, ct.pointer(mat), ct.c_int(first_col), ct.c_int(last_col))

        if err_code:
            raise generate_exception(err_code)

        new_mat = CUDAMatrix(mat)

        try:
            new_mat.sliceof = self.sliceof
        except:
            new_mat.sliceof = self

        return new_mat

    def get_col_slice(self, first_col, last_col, target = None):
        col_slice = self.slice(first_col, last_col)

        if target:
            target.assign(col_slice)
            return target
        else:
            return col_slice

    def set_col_slice(self, first_col, last_col, mat):
        self.slice(first_col, last_col).assign(mat)

        return self

    def get_row_slice(self, start, end, target = None):
        """
        Get the rows with indices start through end. If target is not provided
        memory for a new matrix will be allocated.
        """

        width = self.shape[1]

        if not target:
            target = empty((end-start, width),dtype=self.dtype)

        err_code = _cudamat.get_row_slice(self.p_mat, target.p_mat, ct.c_int(start), ct.c_int(end))
        if err_code:
            raise generate_exception(err_code)

        return target

    def set_row_slice(self, start, end, mat):
        """
        Assign the contents of mat to the rows with indices start through end.
        """

        err_code = _cudamat.set_row_slice(mat.p_mat, self.p_mat, ct.c_int(start), ct.c_int(end))
        if err_code:
            raise generate_exception(err_code)

        return self

    def transpose(self, target = None):
        """
        Return a transposed copy of the matrix.
        """
        if not target:
            target = empty((self.shape[1], self.shape[0]),dtype=self.dtype)

        err_code = _cudamat.copy_transpose(self.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def fill_with_rand(self):
        """
        Fill matrix on the GPU with random numbers drawn from the uniform
        distribution over the (0,1) interval.
        """

        err_code = _cudamat.fill_with_rand(CUDAMatrix.rnd_state_p, self.p_mat) 
        if err_code:
            raise generate_exception(err_code)

        return self

    def fill_with_randn(self):
        """
        Fill matrix on the GPU with random numbers drawn from the standard normal
        distribution.
        """

        err_code = _cudamat.fill_with_randn(CUDAMatrix.rnd_state_p, self.p_mat) 
        if err_code:
            raise generate_exception(err_code)

        return self

    def add_col_vec(self, vec, target = None):
        """
        Add vector vec to every column of the matrix. If a target is provided,
        it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.add_col_vec(self.p_mat, vec.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target
        
    def subtract_col_vec(self, vec, target = None):
        """
        Subtract vector vec to every column of the matrix. If a target is provided,
        it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.sub_col_vec(self.p_mat, vec.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target
        
    def add_col_mult(self, vec, mult, target = None):
        """
        Add a multiple of vector vec to every column of the matrix. If a target
        is provided, it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.add_col_mult(self.p_mat, vec.p_mat, target.p_mat, ct.c_double(mult))
        if err_code:
            raise generate_exception(err_code)

        return target
        
    def add_row_vec(self, vec, target = None):
        """
        Add vector vec to every row of the matrix. If a target is provided,
        it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.add_row_vec(self.p_mat, vec.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target
        
    def subtract_row_vec(self, vec, target = None):
        """
        Subtract vector vec to every row of the matrix. If a target is provided,
        it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.sub_row_vec(self.p_mat, vec.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target
        
    def mult_by_col(self, vec, target = None):
        """
        Multiply vector vec into every column of the matrix. If a target is
        provided, it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.mult_by_col_vec(self.p_mat, vec.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target
        
    def mult_by_row(self, vec, target = None):
        """
        Multiply vector vec into every row of the matrix. If a target is
        provided, it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.mult_by_row_vec(self.p_mat, vec.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target
        
    def sum(self, axis, target = None):
        """
        Sum the matrix along the given dimension, where 0 represents the leading
        dimension and 1 represents the non-leading dimension. If a target is
        not prvided, a new vector is created for storing the result.
        """

        return sum(self, axis, target)

    def add_sums(self, mat, axis, mult = 1.):
        """
        Add a multiple of the sums of the matrix mat along the given dimension
        to self. 
        """

        m = _cudamat.get_leading_dimension(mat.p_mat)
        n = _cudamat.get_nonleading_dimension(mat.p_mat)

        if axis == 0:
            # sum along leading dimension
            left = CUDAMatrix.ones[mat.mat.dtype].slice(0, m)
            left.set_trans(True)
            right = mat
 
        elif axis == 1:
            # sum along non-leading dimension
            left = mat
            right = CUDAMatrix.ones[mat.mat.dtype].slice(0, n)

        err_code = _cudamat.dot(left.p_mat, right.p_mat, self.p_mat, ct.c_double(1.), ct.c_double(mult))
        if err_code:
            raise generate_exception(err_code)

        return self

    def less_than(self, val, target = None):
        """
        Perform the operation target = 1. * (self < val), where val can be a matrix or a scalar.
        """

        if not target:
            target = self

        if isinstance(val, (int, float)):
            err_code = _cudamat.less_than_scalar(self.p_mat, ct.c_double(val), target.p_mat)
        else:
            err_code = _cudamat.less_than(self.p_mat, val.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def greater_than(self, val, target = None):
        """
        Perform the operation target = 1. * (self > val), where val can be a matrix or a scalar.
        """

        if not target:
            target = self

        if isinstance(val, (int, float)):
            err_code = _cudamat.greater_than_scalar(self.p_mat, ct.c_double(val), target.p_mat)
        else:
            err_code = _cudamat.greater_than(self.p_mat, val.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def maximum(self, val, target = None):
        """
        Perform the operation target = maximum(self,val), where val can be a matrix or a scalar.
        """

        if not target:
            target = self

        if isinstance(val, (int, float)):
            err_code = _cudamat.maximum_scalar(self.p_mat, ct.c_double(val), target.p_mat)
        else:
            err_code = _cudamat.maximum(self.p_mat, val.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def max(self, axis, target = None):
        """
        Find the maximum value along the given dimension, where 0 represents the
        leading dimension and 1 represents the non-leading dimension. If a target
        is not prvided, a new vector is created for storing the result.
        """

        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n),dtype=self.dtype)
 
        elif axis == 1:
            if not target:
                target = empty((m, 1),dtype=self.dtype)

        err_code =  _cudamat.max_by_axis(self.p_mat, target.p_mat, ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def min(self, axis, target = None):
        """
        Find the minimum value along the given dimension, where 0 represents the
        leading dimension and 1 represents the non-leading dimension. If a target
        is not prvided, a new vector is created for storing the result.
        """

        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n),dtype=self.dtype)
 
        elif axis == 1:
            if not target:
                target = empty((m, 1),dtype=self.dtype)

        err_code =  _cudamat.min_by_axis(self.p_mat, target.p_mat, ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def sign(self, target = None):
        """
        Find the sign of each element of the matrix.
        """

        if not target:
            target = empty((self.mat.size[0], self.mat.size[1]),dtype=self.dtype)

        err_code = _cudamat.sign(self.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def apply_sigmoid(self, target = None):
        """
        Apply the logistic sigmoid to each element of the matrix.
        """

        return sigmoid(self, target)

    def reciprocal(self, target = None):
        """
        Find the reciprocal of each element of the matrix.
        """

        if not target:
            target = self

        err_code = _cudamat.reciprocal(self.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def diff_cols(self, target = None):
        """
        Compute backward differences along columns
        """
        if not target:
            target = self.slice(1,_cudamat.get_leading_dimension(self.mat.p_mat))

        err_code = _cudamat.diff_cols(self.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def diff_rows(self, target = None):
        """
        Compute backward difference along rows
        """
        if not target:
            raise Exception("not supported")

        err_code = _cudamat.diff_rows(self.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def dot(self, mat2, target = None):
        """
        Multiply the matrix by mat2 from the right.
        """

        return dot(self, mat2, target)

    def add_dot(self, m1, m2):
        """
        Add the dot product of m1 and m2 to the matrix.
        """

        err_code = _cudamat.dot(m1.p_mat, m2.p_mat, self.p_mat, ct.c_double(1.), ct.c_double(1.))
        if err_code:
            raise generate_exception(err_code)

        return self

    def subtract_dot(self, m1, m2):
        """
        Subtract the dot product of m1 and m2 from the matrix.
        """

        err_code = _cudamat.dot(m1.p_mat, m2.p_mat, self.p_mat, ct.c_double(1.), ct.c_double(-1.))
        if err_code:
            raise generate_exception(err_code)

        return self

    def add_mult(self, mat2, alpha = 1.):
        """
        Add multiple of mat2 to the matrix.
        """

        err_code = _cudamat.add_mult(self.p_mat, mat2.p_mat, ct.c_double(alpha))
        if err_code:
            raise generate_exception(err_code)

        return self
    
    def subtract_mult(self, mat2, alpha = 1.):
        """
        Subtract a multiple of mat2 from the matrix.
        """

        err_code = _cudamat.add_mult(self.p_mat, mat2.p_mat, ct.c_double(-1. * alpha))
        if err_code:
            raise generate_exception(err_code)

        return self

    def add(self, val, target = None):
        """Add val to self, where val can be a scalar or a CUDAMatrix with the
        same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            err_code = _cudamat.add_elementwise(self.p_mat, val.p_mat, target.p_mat)
        elif isinstance(val, (int, float)):
            err_code = _cudamat.add_scalar(self.p_mat, ct.c_double(val), target.p_mat)
        else:
            raise ValueError, "Value must be of type CUDAMatrix, int, or float."

        if err_code:
            raise generate_exception(err_code)

        return target


    def add_transpose(self, val, target = None):
        """Add transpose(val) to self, where val must be a CUDAMatrix with
        the same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            err_code = _cudamat.add_transpose(self.p_mat, val.p_mat, target.p_mat)
        else:
            raise ValueError, "Value must be of type CUDAMatrix."

        if err_code:
            raise generate_exception(err_code)

        return target

    def subtract(self, val, target = None):
        """Subtract val from self, where val can be a scalar or a CUDAMatrix with
        the same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            err_code = _cudamat.subtract_elementwise(self.p_mat, val.p_mat, target.p_mat)
        elif isinstance(val, (int, float)):
            err_code = _cudamat.add_scalar(self.p_mat, ct.c_double(-1*val), target.p_mat)
        else:
            raise ValueError, "Value must be of type CUDAMatrix, int, or float."

        if err_code:
            raise generate_exception(err_code)

        return target

    def subtract_transpose(self, val, target = None):
        """Subtract transpose(val) from self, where val must be a CUDAMatrix with
        the same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            err_code = _cudamat.subtract_transpose(self.p_mat, val.p_mat, target.p_mat)
        else:
            raise ValueError, "Value must be of type CUDAMatrix."

        if err_code:
            raise generate_exception(err_code)

        return target

    def divide(self, val, target = None):
        """Divide self by val, where val can be a scalar or a CUDAMatrix with the
        same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            err_code = _cudamat.divide_elementwise(self.p_mat, val.p_mat, target.p_mat)
        elif isinstance(val, (int, float)):
            err_code = _cudamat.divide_by_scalar(self.p_mat, ct.c_double(val), target.p_mat)
        else:
            raise ValueError, "Value must be of type CUDAMatrix, int, or float."

        if err_code:
            raise generate_exception(err_code)

        return target

    def mult(self, val, target = None):
        """Multiply self by val, where val can be a scalar or a CUDAMatrix with
        the same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            err_code = _cudamat.mult_elementwise(self.p_mat, val.p_mat, target.p_mat)
        elif isinstance(val, (int, float)):
            err_code = _cudamat.mult_by_scalar(self.p_mat, ct.c_double(val), target.p_mat)
        else:
            raise ValueError, "Value must be of type CUDAMatrix, int, or float."

        if err_code:
            raise generate_exception(err_code)

        return target

    def clip_norm(self, vec, eps, target = None):
        """
        Multiply each column of self elementwise by the reciprocal square root of vec;
        however, if vec[i] < eps, then self[i] is not changed. If a target is
        provided, it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.clip_norm(self.p_mat, vec.p_mat, ct.c_double(eps), target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    @deprecated
    def assign_scalar(self, alpha):
        """
        Assign scalar alpha to every element of the matrix.
        """

        err_code = _cudamat.assign_scalar(self.p_mat, ct.c_double(alpha))
        if err_code:
            raise generate_exception(err_code)

        return self

    @deprecated
    def mult_by_scalar(self, alpha, target = None):
        """
        Multiply the matrix by a scalar.
        """

        if not target:
            target = self

        err_code = _cudamat.mult_by_scalar(self.p_mat, ct.c_double(alpha), target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target


    @deprecated
    def div_by_scalar(self, alpha, target = None):
        """
        Divide the matrix by a scalar.
        """

        if not target:
            target = self

        err_code = _cudamat.divide_by_scalar(self.p_mat, ct.c_double(alpha), target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    @deprecated
    def add_scalar(self, alpha, target = None):
        """
        Increment the matrix by a scalar.
        """

        if not target:
            target = self

        err_code = _cudamat.add_scalar(self.p_mat, ct.c_double(alpha), target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def euclid_norm(self):
        err_code = ct.c_int(0)
        res = _cudamat.euclid_norm(self.p_mat, ct.byref(err_code))

        if err_code:
            raise generate_exception(err_code.value)

        return res

    def select_columns(self, indices, target):
        """
        copies some columns of self into target.
        <indices> must be a row vector. Its elements are float32's representing integers, e.g. "34.0" means the integer "34".
        after this call, for all r,c, target[r,c]=self[r,indices[c]].
        This returns target.
        Negative indices are interpreted in the usual Python way: all elements of <indices> had better be in the range [-self.shape[1], self.shape[1]-1].
        This does bounds checking, but out of bounds indices do not raise an exception (because the programmer was lazy). Instead, they result in NaN values in <target>.
        """

        err_code = _cudamat.selectRows(self.p_mat, target.p_mat, indices.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def set_selected_columns(self, indices, source):
        """
        copies all columns of source into some columns of self.
        <indices> must be a row vector. Its elements are float32's representing
        integers, e.g. "34.0" means the integer "34". after this call, for all
        r,c, self[r,indices[c]]=source[r,c]. This returns self.
        Negative indices are interpreted in the usual Python way: all elements
        of <indices> had better be in the range [-self.shape[1], self.shape[1]-1].
        This does bounds checking, but out of bounds indices do not raise an
        exception (because the programmer was lazy). Instead, they result in NaN
        values in <self>.
        """

        err_code = _cudamat.setSelectedRows(self.p_mat, source.p_mat, indices.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return self

def empty(shape,dtype='float32'):
    """
    Creates and returns a new CUDAMatrix with the given shape.
    """

    mat = cudamat()
    err_code = _cudamat.init_empty(ct.pointer(mat), ct.c_int(shape[0]), ct.c_int(shape[1]), ct.c_int(dtype_np2cm(dtype)))

    if err_code:
        raise generate_exception(err_code)

    return CUDAMatrix(mat)

def sum(mat, axis, target = None):
    """
    Sum the matrix along the given dimension, where 0 represents the leading
    dimension and 1 represents the non-leading dimension. If a target is
    not prvided, a new vector is created for storing the result.
    """

    m = _cudamat.get_leading_dimension(mat.p_mat)
    n = _cudamat.get_nonleading_dimension(mat.p_mat)

    if axis == 0:
        # sum along leading dimension
        left = CUDAMatrix.ones[mat.mat.dtype].slice(0, m)
        left.set_trans(True)
        right = mat

        if not target:
            target = empty((1, n),dtype=mat.dtype)
 
    elif axis == 1:
        # sum along non-leading dimension
        left = mat
        right = CUDAMatrix.ones[mat.mat.dtype].slice(0, n)

        if not target:
            target = empty((m, 1),dtype=mat.dtype)

    err_code = _cudamat.dot(left.p_mat, right.p_mat, target.p_mat, ct.c_double(0.), ct.c_double(1.))
    if err_code:
        raise generate_exception(err_code)

    return target

def dot(m1, m2, target = None):
    """
    Find the dot product between m1 and m2.
    """

    if not target:
        m = _cudamat.get_leading_dimension(m1.p_mat)
        n = _cudamat.get_nonleading_dimension(m2.p_mat)

        target = empty((m, n),dtype=m1.dtype)

    err_code = _cudamat.dot(m1.p_mat, m2.p_mat, target.p_mat, ct.c_double(0.), ct.c_double(1.))
    if err_code:
        raise generate_exception(err_code)

    return target

def vdot(m1, m2):
    """
    Compute the vector dot product of matrices m1 and m2.
    """

    err_code = ct.c_int(0)
    res = _cudamat.vdot(m1.p_mat, m2.p_mat, ct.byref(err_code))

    if err_code:
        raise generate_exception(err_code.value)

    return res

def relu(mat, target = None, dtarget = None):
    """
    Apply the logistic sigmoid to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_relu(mat.p_mat, target.p_mat, dtarget.p_mat if dtarget != None else None)
    if err_code:
        raise generate_exception(err_code)

    return target

def sigmoid(mat, target = None):
    """
    Apply the logistic sigmoid to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_sigmoid(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def tanh(mat, target = None):
    """
    Apply the tanh to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_tanh(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def sigmoid_deriv(mat, target = None):
    """
    Apply sigmoid'(x) to each element of the matrix mat, assuming mat contains sigmoid(x)
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_sigmoid_deriv(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def tanh_deriv(mat, target = None):
    """
    Apply tanh'(x) to each element of the matrix mat, assuming mat contains tanh(x)
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_tanh_deriv(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def abs(mat, target = None):
    """
    Apply abs to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_abs(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def log_1_plus_exp(mat, target = None):
    """
    Apply log(1+exp(x)) to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_log_1_plus_exp(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def log(mat, target = None):
    """
    Find the natural logarithm of each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_log(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def exp(mat, target = None):
    """
    Apply the exponential function to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_exp(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def sqrt(mat, target = None):
    """
    Compute the square root of each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_sqrt(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def pow(mat, p, target = None):
    """
    If p is a scalar, compute the 'p'th power of each element of the matrix mat,
    otherwise raise each element of the matrix mat to the power given by the
    corresponding element of the matrix p.
    """

    if not target:
        target = mat

    if isinstance(p, CUDAMatrix):
        err_code = _cudamat.apply_pow_matrix(mat.p_mat, p.p_mat, target.p_mat)
    elif isinstance(p, (int, float)):
        err_code = _cudamat.apply_pow(mat.p_mat, ct.c_double(p), target.p_mat)
    else:
        raise ValueError, "Value must be of type CUDAMatrix, int, or float."

    if err_code:
        raise generate_exception(err_code)

    return target

def square(mat, target = None):
    """
    Square each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.square(mat.p_mat, target.p_mat)

    if err_code:
        raise generate_exception(err_code)

    return target

def dropout(matA, matB, dropout_rate, targetA = None, targetB = None):
    """
    Set each element of mat to zero, independently, with probability "dropout_rate".
    """

    if not targetA:
        targetA = matA

    if not targetB:
        targetB = matB

    if matB != None:
        err_code = _cudamat.dropout(CUDAMatrix.rnd_state_p, 
                                    matA.p_mat, matB.p_mat,
                                    ct.c_float(dropout_rate), 
                                    targetA.p_mat,targetB.p_mat)
    else:
        err_code = _cudamat.dropout(CUDAMatrix.rnd_state_p, 
                                    matA.p_mat, None,
                                    ct.c_float(dropout_rate), 
                                    targetA.p_mat,None)

    if err_code:
        raise generate_exception(err_code)


def cuda_sync_threads():
    _cudamat.cuda_sync_threads()

def reformat(array):
    """
    Returns array in FORTRAN order.
    """

    return np.array(array, order='F')

def cuda_set_device(dev_id):
    """
    Selects the CUDA device with the given ID.
    """

    err_code =  _cudamat.cuda_set_device(ct.c_int(dev_id))
    if err_code:
        raise generate_exception(err_code)

def cuda_get_device():
    """
    Returns the ID of the currently selected CUDA device.
    """

    device_id = ct.c_int(-1)
    err_code =  _cudamat.cuda_get_device(ct.byref(device_id))
    if err_code:
        raise generate_exception(err_code)
    return device_id.value

def cuda_device_reset():
    err_code = _cudamat.cuda_device_reset()
    if err_code:
        raise generate_exception(err_code)

def cuda_get_device_count():
    return _cudamat.cuda_get_device_count()

def cuda_get_device_prop(device):
    prop = CudaDeviceProp()
    err_code = _cudamat.cuda_get_device_prop(ct.byref(prop),device)
    if err_code:
        raise generate_exception(err_code)
    return prop


def cuda_memory_info():
    """
    Returns (available memory, total memory) in bytes
    """
    avail = _cudamat.cuda_memory_available()
    total = _cudamat.cuda_memory_total()
    return (avail,total)

def cublas_init():
    """
    Initialize Cublas.
    """
    global supported_dtypes
    err_code = _cudamat.cublas_init()
    if err_code:
        raise generate_exception(err_code)
    CUDAMatrix.ones = []
    for i in range(len(supported_dtypes)):
        CUDAMatrix.ones.append(CUDAMatrix(np.ones((MAX_ONES, 1), dtype=dtype_cm2np(i), order = 'F')))

init = cublas_init

def cublas_shutdown():
    """
    Shut down Cublas.
    """

    CUDAMatrix.ones = 0
    _cudamat.cublas_shutdown()

shutdown = cublas_shutdown
