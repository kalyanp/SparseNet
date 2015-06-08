#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include <cuda.h>
#include <limits>
//#include "cudamat_kernels.cuh"
#include "cudamat_a.cuh"
#include "cudamat_kernels.cu"

//#define AUTO_CUDA_SYNC
#ifdef  AUTO_CUDA_SYNC
#define CUDA_THREAD_SYNC() cudaThreadSynchronize();
#else
#define CUDA_THREAD_SYNC() 
#endif

#ifdef _MSC_VER
#define DLLEXP __declspec(dllexport)
#else
#define DLLEXP
#endif

extern "C" {

typedef unsigned char ubyte;
typedef double        doubl;
typedef unsigned int  unsig;

inline int dtype_size(int dtype)
{
	if (dtype == 0) return sizeof(float);
	if (dtype == 1) return sizeof(double);
	if (dtype == 2) return sizeof(unsigned char);
	if (dtype == 3) return sizeof(unsigned int);
	return -1;
}

/* ------------------------------ CUBLAS init/shutdown ------------------------------ */

inline bool check_cublas_error() {
    cublasStatus status = cublasGetError();

    return status != CUBLAS_STATUS_SUCCESS;
}

inline bool checkCUDAError(cudaError_t err = cudaSuccess) {
	if (err == cudaSuccess)
		err = cudaGetLastError();

    if (err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));
    return cudaSuccess != err;
}

DLLEXP extern const char* get_last_cuda_error() {
    cudaError_t err = cudaGetLastError();

    return cudaGetErrorString( err);
}

DLLEXP extern int cublas_init() {
    cublasInit();
    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

DLLEXP extern int cublas_shutdown() {
    cublasShutdown();
    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

DLLEXP extern void cuda_device_reset() {
    checkCUDAError(cudaDeviceReset());
}

DLLEXP extern int cuda_get_device(int* device_id)
{
	cudaError_t error = cudaGetDevice(device_id);
    return error? CUDA_ERROR : 0;
}

DLLEXP extern int cuda_get_device_count()
{
	int count = 0;
	cudaGetDeviceCount(&count);
	return count;
}

DLLEXP extern int cuda_get_device_prop(cudaDeviceProp* prop, int device)
{
	cudaError_t error = cudaGetDeviceProperties(prop,device);
	return error ? CUDA_ERROR : 0;
}

DLLEXP extern size_t cuda_memory_available()
{
	// force device to be ready for cuMemGetInfo
	void* ptr = 0;
	cudaMalloc(&ptr,128);
	cudaFree(ptr);

	size_t free = 0, total = 0;
	CUresult err = cuMemGetInfo(&free,&total);
	return free;
}

DLLEXP extern size_t cuda_memory_total()
{
	// force device to be ready for cuMemGetInfo
	void* ptr = 0;
	cudaMalloc(&ptr,128);
	cudaFree(ptr);

	size_t free = 0, total = 0;
	CUresult err = cuMemGetInfo(&free,&total);
	return total;
}

DLLEXP extern int cuda_set_device(int deviceId) {
	cudaError_t error = cudaSetDevice(deviceId);
	return checkCUDAError(error) ? CUDA_ERROR : 0;
}

DLLEXP extern int init_random(rnd_struct* rnd_state, int seed, char* cudamatpath) {
    unsigned int * host_mults;
    host_mults = (unsigned int*)malloc(NUM_RND_STREAMS * sizeof(unsigned int));
    FILE * pFile;

    pFile = fopen (cudamatpath,"r");

    for (int i = 0; i < NUM_RND_STREAMS; i++) {
        fscanf (pFile, "%u", &host_mults[i]);
    }
    fclose (pFile);

    cublasAlloc(NUM_RND_STREAMS, sizeof(unsigned int), (void**)&rnd_state->dev_mults);
    cublasAlloc(NUM_RND_STREAMS, sizeof(unsigned long long), (void**)&rnd_state->dev_words);
    cublasSetVector(NUM_RND_STREAMS, sizeof(unsigned int), host_mults, 1, rnd_state->dev_mults, 1);
    //cudaMalloc((void **)&rnd_state->dev_mults, NUM_RND_STREAMS * sizeof(unsigned int));
    //cudaMalloc((void **)&rnd_state->dev_words, NUM_RND_STREAMS * sizeof(unsigned long long));
    //cudaMemcpy(rnd_state->dev_mults, host_mults, NUM_RND_STREAMS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    kSeedRandom<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, seed);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

/* ------------------------------ Utility routines ------------------------------ */

DLLEXP extern int get_leading_dimension(cudamat* mat) {
    return mat->is_trans ? mat->size[1] : mat->size[0];
}

DLLEXP extern int get_nonleading_dimension(cudamat* mat) {
    return mat->is_trans ? mat->size[0] : mat->size[1];
}

DLLEXP extern void set_transpose(cudamat* mat, int is_trans) {
    mat->is_trans = is_trans;
}

inline char get_transpose_char(cudamat* mat) {
    return mat->is_trans ? 't' : 'n';
}

DLLEXP extern void cuda_sync_threads() {
    cudaThreadSynchronize();
}

/* ------------------------------ Allocating/moving data ------------------------------ */

DLLEXP extern int allocate_device_memory(cudamat* mat) {
    int len = mat->size[0]*mat->size[1];

    cublasStatus stat;

	if (dtype_size(mat->dtype) <= 0)
		return ERROR_DTYPE_UNSUPPORTED;

    stat = cublasAlloc(len, dtype_size(mat->dtype), &mat->data_device);

    if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error())
        return CUBLAS_ERROR;

    mat->on_device = 1;
    return 0;
}

DLLEXP extern int copy_to_host(cudamat* mat) {
    int len = mat->size[0]*mat->size[1];

    if (mat->on_device) {
        cublasGetVector(len, dtype_size(mat->dtype), mat->data_device, 1, mat->data_host, 1);

        if (check_cublas_error())
            return CUBLAS_ERROR;
    } else
       return ERROR_NOT_ON_DEVICE;
 
    return 0;
}

DLLEXP extern int copy_to_device(cudamat* mat) {
    int len = mat->size[0]*mat->size[1];
    int err_code = 0;

    //if (!mat->owns_data)
    //    return VIEW_ERROR;

    if (!mat->on_device) {
        err_code = allocate_device_memory(mat);
        if (err_code)
            return err_code;
    }

    cublasSetVector(len, dtype_size(mat->dtype), mat->data_host, 1, mat->data_device, 1);
    
    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}

DLLEXP extern int copy_on_device(cudamat* mat1, cudamat* mat2) {
    int len = mat1->size[0]*mat1->size[1];

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat1->dtype != mat2->dtype)
        return ERROR_MISMATCHED_DTYPE;

	//cublasScopy(len, mat1->data_device, 1, mat2->data_device, 1);
	cudaMemcpy(mat2->data_device,mat1->data_device,len*dtype_size(mat1->dtype),cudaMemcpyDeviceToDevice);

    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

DLLEXP extern int get_row_slice(cudamat* source, cudamat* target, unsigned int start, unsigned int end) {
    unsigned int height = source->size[0];
    unsigned int width = source->size[1];

    if ((end - start) != target->size[0] || source->size[1] != target->size[1] || start >= end || end > height)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (source->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    dim3 kernelBlockGrid((int)ceil((end - start)/32.), (int)ceil(width/32.), 1);
    dim3 kernelBlockDim(32, 1, 1);

	if (source->dtype == 0)  kGetRowSlice<float><<<kernelBlockGrid,kernelBlockDim>>>((float*)source->data_device, (float*)target->data_device, start, end, width, height);
	if (source->dtype == 1)  kGetRowSlice<doubl><<<kernelBlockGrid,kernelBlockDim>>>((doubl*)source->data_device, (doubl*)target->data_device, start, end, width, height);
	if (source->dtype == 2)  kGetRowSlice<ubyte><<<kernelBlockGrid,kernelBlockDim>>>((ubyte*)source->data_device, (ubyte*)target->data_device, start, end, width, height);
	if (source->dtype == 3)  kGetRowSlice<unsig><<<kernelBlockGrid,kernelBlockDim>>>((unsig*)source->data_device, (unsig*)target->data_device, start, end, width, height);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

DLLEXP extern int set_row_slice(cudamat* source, cudamat* target,  unsigned int start, unsigned int end) {
    unsigned int height = target->size[0];
    unsigned int width = target->size[1];

    if ((end - start) != source->size[0] || source->size[1] != target->size[1] || start >= end || end > height)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (source->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    dim3 kernelBlockGrid((int)ceil((end - start)/32.), (int)ceil(width/32.), 1);
    dim3 kernelBlockDim(32, 1, 1);

	if (source->dtype == 0)  kSetRowSlice<float><<<kernelBlockGrid,kernelBlockDim>>>((float*)source->data_device, (float*)target->data_device, start, end, width, height);
	if (source->dtype == 1)  kSetRowSlice<doubl><<<kernelBlockGrid,kernelBlockDim>>>((doubl*)source->data_device, (doubl*)target->data_device, start, end, width, height);
	if (source->dtype == 2)  kSetRowSlice<ubyte><<<kernelBlockGrid,kernelBlockDim>>>((ubyte*)source->data_device, (ubyte*)target->data_device, start, end, width, height);
	if (source->dtype == 3)  kSetRowSlice<unsig><<<kernelBlockGrid,kernelBlockDim>>>((unsig*)source->data_device, (unsig*)target->data_device, start, end, width, height);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

DLLEXP extern int copy_transpose(cudamat* source, cudamat* target) {
    unsigned int height = source->size[0];
    unsigned int width = source->size[1];

    if (source->size[0] != target->size[1] || source->size[1] != target->size[0])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (source->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    // setup execution parameters
    unsigned int grid_x = height / COPY_BLOCK_SIZE;
    if (height % COPY_BLOCK_SIZE)
        grid_x++;

    unsigned int grid_y = width / COPY_BLOCK_SIZE;
    if (width % COPY_BLOCK_SIZE)
        grid_y++;

    dim3 grid(grid_x, grid_y, 1);
    dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);

    if (source->dtype == 0)  kTranspose<float><<< grid, threads >>>((float*)target->data_device, (float*)source->data_device, height, width);
    if (source->dtype == 1)  kTranspose<doubl><<< grid, threads >>>((doubl*)target->data_device, (doubl*)source->data_device, height, width);
    if (source->dtype == 2)  kTranspose<ubyte><<< grid, threads >>>((ubyte*)target->data_device, (ubyte*)source->data_device, height, width);
    if (source->dtype == 3)  kTranspose<unsig><<< grid, threads >>>((unsig*)target->data_device, (unsig*)source->data_device, height, width);

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

DLLEXP extern int free_device_memory(cudamat* mat) {
    if (mat->owns_data && mat->on_device) {
        cublasStatus stat;

        stat = cublasFree(mat->data_device);
        mat->on_device = 0;

        if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error())
            return CUBLAS_ERROR;
    }

    return 0;
}

DLLEXP extern int reshape(cudamat* mat, unsigned int m, unsigned int n) {
    if (mat->size[0] * mat->size[1] != m * n)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->size[0] = m;
    mat->size[1] = n;

    return 0;
}

DLLEXP extern int get_slice(cudamat* source, cudamat* target, unsigned int first_col, unsigned int last_col) {
    if (source->is_trans)
        return ERROR_TRANSPOSED;
    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (last_col > source->size[1] || (first_col >= last_col))
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    int num_rows = source->size[0];

    target->data_host = 0;
    target->data_device = (unsigned char*)source->data_device + first_col * num_rows * dtype_size(source->dtype);
    target->on_device = 1;
    target->on_host = 0;
    target->size[0] = source->size[0];
    target->size[1] = last_col - first_col;
    target->is_trans = 0;
    target->owns_data = 0;
	target->dtype = source->dtype;

    return 0;
}

DLLEXP extern int get_vector_slice(cudamat* source, cudamat* target, unsigned int first_ind, unsigned int last_ind) {
    // source must be a vector
    if (source->size[0] > 1 && source->size[1] > 1)
        return ERROR_GENERIC;
    if (source->is_trans)
        return ERROR_TRANSPOSED;
    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (first_ind >= last_ind)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    unsigned int num_rows = source->size[0];

    target->data_host = 0;
    target->data_device = (unsigned char*)source->data_device + first_ind * num_rows * dtype_size(source->dtype);
    target->on_device = 1;
    target->on_host = 0;
    target->is_trans = 0;
    target->owns_data = 0;
	target->dtype = source->dtype;

    if (source->size[0] > 1) {
        if (last_ind > source->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        target->size[0] = last_ind - first_ind;
        target->size[1] = 1;
    } else {
        if (last_ind > source->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        target->size[0] = 1;
        target->size[1] = last_ind - first_ind;
    }

    return 0;
}

/* ------------------------------ Initialization routines ------------------------------ */

DLLEXP extern void init_from_array(cudamat* mat, void* data, int m, int n, int dtype) {
    mat->data_host = data;
	mat->dtype = dtype;
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 0;
    mat->on_host = 1;
    mat->is_trans = 0;
    mat->owns_data = 1;
}

DLLEXP extern int init_empty(cudamat* mat, int m, int n, int dtype) {
	mat->dtype = dtype;
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 0;
    mat->on_host = 0;
    mat->is_trans = 0;
    mat->owns_data = 1;

    return allocate_device_memory(mat);
}

/* ------------------------------ Random number generation ------------------------------ */
DLLEXP extern int fill_with_rand(rnd_struct* rnd_state, cudamat* mat) {
    int len = mat->size[0] * mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->dtype != 0)
        return ERROR_DTYPE_UNSUPPORTED;

    kRandomUniform<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, (float*)mat->data_device, len);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

DLLEXP extern int fill_with_randn(rnd_struct* rnd_state, cudamat* mat) {
    int len = mat->size[0] * mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->dtype != 0)
        return ERROR_DTYPE_UNSUPPORTED;

    kRandomGaussian<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, (float*)mat->data_device, len);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}
/* ------------------------------ Algebraic operations ------------------------------ */

DLLEXP extern int diff_cols(cudamat* mat, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans)
        return ERROR_TRANSPOSED;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1]+1)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    if (mat->dtype == 0) kDiffCols<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, w, h);
    if (mat->dtype == 1) kDiffCols<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, w, h);
	if (mat->dtype == 2) kDiffCols<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte*)target->data_device, w, h);
	if (mat->dtype == 3) kDiffCols<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig*)target->data_device, w, h);

    CUDA_THREAD_SYNC();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}

DLLEXP extern int diff_rows(cudamat* mat, cudamat* target) {
    return ERROR_UNSUPPORTED; // TODO
}

DLLEXP extern int add_col_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans)
        return ERROR_TRANSPOSED;
    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != vec->dtype || mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    if (mat->dtype == 0) kAddColVector<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)vec->data_device, (float*)target->data_device, w, h);
    if (mat->dtype == 1) kAddColVector<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)vec->data_device, (doubl*)target->data_device, w, h);
	if (mat->dtype == 2) kAddColVector<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte*)vec->data_device, (ubyte*)target->data_device, w, h);
	if (mat->dtype == 3) kAddColVector<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig*)vec->data_device, (unsig*)target->data_device, w, h);

    CUDA_THREAD_SYNC();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}

DLLEXP extern int sub_col_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans)
        return ERROR_TRANSPOSED;
    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != vec->dtype || mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    if (mat->dtype == 0) kSubColVector<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)vec->data_device, (float*)target->data_device, w, h);
    if (mat->dtype == 1) kSubColVector<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)vec->data_device, (doubl*)target->data_device, w, h);
	if (mat->dtype == 2) kSubColVector<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte*)vec->data_device, (ubyte*)target->data_device, w, h);
	if (mat->dtype == 3) kSubColVector<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig*)vec->data_device, (unsig*)target->data_device, w, h);

    CUDA_THREAD_SYNC();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}

DLLEXP extern int add_col_mult(cudamat* mat, cudamat* vec, cudamat* target, double mult) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans)
        return ERROR_TRANSPOSED;
    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != vec->dtype || mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    if (mat->dtype == 0) kAddColMult<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)vec->data_device, (float*)target->data_device, (float)mult, w, h);
    if (mat->dtype == 1) kAddColMult<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)vec->data_device, (doubl*)target->data_device, (doubl)mult, w, h);
	if (mat->dtype == 2) kAddColMult<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte*)vec->data_device, (ubyte*)target->data_device, (ubyte)mult, w, h);
	if (mat->dtype == 3) kAddColMult<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig*)vec->data_device, (unsig*)target->data_device, (unsig)mult, w, h);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int add_row_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans)
        return ERROR_TRANSPOSED;
    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != vec->dtype || mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    if (mat->dtype == 0) kAddRowVector<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)vec->data_device, (float*)target->data_device, w, h);
    if (mat->dtype == 1) kAddRowVector<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)vec->data_device, (doubl*)target->data_device, w, h);
	if (mat->dtype == 2) kAddRowVector<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte*)vec->data_device, (ubyte*)target->data_device, w, h);
	if (mat->dtype == 3) kAddRowVector<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig*)vec->data_device, (unsig*)target->data_device, w, h);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int sub_row_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans)
        return ERROR_TRANSPOSED;
    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != vec->dtype || mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    if (mat->dtype == 0) kSubRowVector<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)vec->data_device, (float*)target->data_device, w, h);
    if (mat->dtype == 1) kSubRowVector<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)vec->data_device, (doubl*)target->data_device, w, h);
	if (mat->dtype == 2) kSubRowVector<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte*)vec->data_device, (ubyte*)target->data_device, w, h);
	if (mat->dtype == 3) kSubRowVector<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig*)vec->data_device, (unsig*)target->data_device, w, h);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int mult_by_col_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans)
        return ERROR_TRANSPOSED;
    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != vec->dtype || mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    if (mat->dtype == 0) kMultByColVector<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)vec->data_device, (float*)target->data_device, w, h);
    if (mat->dtype == 1) kMultByColVector<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)vec->data_device, (doubl*)target->data_device, w, h);
	if (mat->dtype == 2) kMultByColVector<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte*)vec->data_device, (ubyte*)target->data_device, w, h);
	if (mat->dtype == 3) kMultByColVector<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig*)vec->data_device, (unsig*)target->data_device, w, h);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int mult_by_row_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans)
        return ERROR_TRANSPOSED;
    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != vec->dtype || mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    if (mat->dtype == 0) kMultByRowVector<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)vec->data_device, (float*)target->data_device, w, h);
    if (mat->dtype == 1) kMultByRowVector<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)vec->data_device, (doubl*)target->data_device, w, h);
	if (mat->dtype == 2) kMultByRowVector<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte*)vec->data_device, (ubyte*)target->data_device, w, h);
	if (mat->dtype == 3) kMultByRowVector<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig*)vec->data_device, (unsig*)target->data_device, w, h);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int clip_norm(cudamat* mat, cudamat* vec, double eps, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans)
        return ERROR_TRANSPOSED;
    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != vec->dtype || mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    if (mat->dtype == 0) kClipNorm<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)vec->data_device, (float)eps, (float*)target->data_device, w, h);
    if (mat->dtype == 1) kClipNorm<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)vec->data_device, (doubl)eps, (doubl*)target->data_device, w, h);
	if (mat->dtype == 2) kClipNorm<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte*)vec->data_device, (ubyte)eps, (ubyte*)target->data_device, w, h);
	if (mat->dtype == 3) kClipNorm<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig*)vec->data_device, (unsig)eps, (unsig*)target->data_device, w, h);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int less_than(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat1->dtype != mat2->dtype)
        return ERROR_MISMATCHED_DTYPE;

	if (target->dtype == 0) {
		if (mat1->dtype == 0)  kLessThan<float,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat1->data_device, (float*)mat2->data_device, (float*)target->data_device, len);
		if (mat1->dtype == 1)  kLessThan<doubl,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (float*)target->data_device, len);
		if (mat1->dtype == 2)  kLessThan<ubyte,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (float*)target->data_device, len);
		if (mat1->dtype == 3)  kLessThan<unsig,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (float*)target->data_device, len);
	} else if (target->dtype == 2) {
		if (mat1->dtype == 0)  kLessThan<float,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat1->data_device, (float*)mat2->data_device, (ubyte*)target->data_device, len);
		if (mat1->dtype == 1)  kLessThan<doubl,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (ubyte*)target->data_device, len);
		if (mat1->dtype == 2)  kLessThan<ubyte,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (ubyte*)target->data_device, len);
		if (mat1->dtype == 3)  kLessThan<unsig,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (ubyte*)target->data_device, len);
	} else if (target->dtype == 3) {
		if (mat1->dtype == 0)  kLessThan<float,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat1->data_device, (float*)mat2->data_device, (unsig*)target->data_device, len);
		if (mat1->dtype == 1)  kLessThan<doubl,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (unsig*)target->data_device, len);
		if (mat1->dtype == 2)  kLessThan<ubyte,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (unsig*)target->data_device, len);
		if (mat1->dtype == 3)  kLessThan<unsig,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (unsig*)target->data_device, len);
	} else {
		return ERROR_DTYPE_UNSUPPORTED;
	}

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int less_than_scalar(cudamat* mat, double val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

	if (target->dtype == 0) {
		if (mat->dtype == 0)  kLessThanScalar<float,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float)val, (float*)target->data_device, len);
		if (mat->dtype == 1)  kLessThanScalar<doubl,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl)val, (float*)target->data_device, len);
		if (mat->dtype == 2)  kLessThanScalar<ubyte,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte)val, (float*)target->data_device, len);
		if (mat->dtype == 3)  kLessThanScalar<unsig,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig)val, (float*)target->data_device, len);
	} else if (target->dtype == 2) {
		if (mat->dtype == 0)  kLessThanScalar<float,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float)val, (ubyte*)target->data_device, len);
		if (mat->dtype == 1)  kLessThanScalar<doubl,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl)val, (ubyte*)target->data_device, len);
		if (mat->dtype == 2)  kLessThanScalar<ubyte,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte)val, (ubyte*)target->data_device, len);
		if (mat->dtype == 3)  kLessThanScalar<unsig,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig)val, (ubyte*)target->data_device, len);
	} else if (target->dtype == 3) {
		if (mat->dtype == 0)  kLessThanScalar<float,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float)val, (unsig*)target->data_device, len);
		if (mat->dtype == 1)  kLessThanScalar<doubl,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl)val, (unsig*)target->data_device, len);
		if (mat->dtype == 2)  kLessThanScalar<ubyte,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte)val, (unsig*)target->data_device, len);
		if (mat->dtype == 3)  kLessThanScalar<unsig,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig)val, (unsig*)target->data_device, len);
	} else {
		return ERROR_DTYPE_UNSUPPORTED;
	}

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int greater_than(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat1->dtype != mat2->dtype)
        return ERROR_MISMATCHED_DTYPE;

	if (target->dtype == 0) {
		if (mat1->dtype == 0)  kGreaterThan<float,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat1->data_device, (float*)mat2->data_device, (float*)target->data_device, len);
		if (mat1->dtype == 1)  kGreaterThan<doubl,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (float*)target->data_device, len);
		if (mat1->dtype == 2)  kGreaterThan<ubyte,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (float*)target->data_device, len);
		if (mat1->dtype == 3)  kGreaterThan<unsig,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (float*)target->data_device, len);
	} else if (target->dtype == 2) {
		if (mat1->dtype == 0)  kGreaterThan<float,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat1->data_device, (float*)mat2->data_device, (ubyte*)target->data_device, len);
		if (mat1->dtype == 1)  kGreaterThan<doubl,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (ubyte*)target->data_device, len);
		if (mat1->dtype == 2)  kGreaterThan<ubyte,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (ubyte*)target->data_device, len);
		if (mat1->dtype == 3)  kGreaterThan<unsig,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (ubyte*)target->data_device, len);
	} else if (target->dtype == 3) {
		if (mat1->dtype == 0)  kGreaterThan<float,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat1->data_device, (float*)mat2->data_device, (unsig*)target->data_device, len);
		if (mat1->dtype == 1)  kGreaterThan<doubl,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (unsig*)target->data_device, len);
		if (mat1->dtype == 2)  kGreaterThan<ubyte,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (unsig*)target->data_device, len);
		if (mat1->dtype == 3)  kGreaterThan<unsig,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (unsig*)target->data_device, len);
	} else {
		return ERROR_DTYPE_UNSUPPORTED;
	}
    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int greater_than_scalar(cudamat* mat, double val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

	if (target->dtype == 0) {
		if (mat->dtype == 0)  kGreaterThanScalar<float,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float)val, (float*)target->data_device, len);
		if (mat->dtype == 1)  kGreaterThanScalar<doubl,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl)val, (float*)target->data_device, len);
		if (mat->dtype == 2)  kGreaterThanScalar<ubyte,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte)val, (float*)target->data_device, len);
		if (mat->dtype == 3)  kGreaterThanScalar<unsig,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig)val, (float*)target->data_device, len);
	} else if (target->dtype == 2) {
		if (mat->dtype == 0)  kGreaterThanScalar<float,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float)val, (ubyte*)target->data_device, len);
		if (mat->dtype == 1)  kGreaterThanScalar<doubl,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl)val, (ubyte*)target->data_device, len);
		if (mat->dtype == 2)  kGreaterThanScalar<ubyte,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte)val, (ubyte*)target->data_device, len);
		if (mat->dtype == 3)  kGreaterThanScalar<unsig,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig)val, (ubyte*)target->data_device, len);
	} else if (target->dtype == 3) {
		if (mat->dtype == 0)  kGreaterThanScalar<float,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float)val, (unsig*)target->data_device, len);
		if (mat->dtype == 1)  kGreaterThanScalar<doubl,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl)val, (unsig*)target->data_device, len);
		if (mat->dtype == 2)  kGreaterThanScalar<ubyte,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte)val, (unsig*)target->data_device, len);
		if (mat->dtype == 3)  kGreaterThanScalar<unsig,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig)val, (unsig*)target->data_device, len);
	} else {
		return ERROR_DTYPE_UNSUPPORTED;
	}
   CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int maximum(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat1->dtype != mat2->dtype || mat1->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	if (mat1->dtype == 0) kMaximum<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat1->data_device, (float*)mat2->data_device, (float*)target->data_device, len);
	if (mat1->dtype == 1) kMaximum<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (doubl*)target->data_device, len);
	if (mat1->dtype == 2) kMaximum<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (ubyte*)target->data_device, len);
	if (mat1->dtype == 3) kMaximum<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (unsig*)target->data_device, len);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int maximum_scalar(cudamat* mat, double val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	if (mat->dtype == 0) kMaximumScalar<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float)val, (float*)target->data_device, len);
	if (mat->dtype == 1) kMaximumScalar<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl)val, (doubl*)target->data_device, len);
	if (mat->dtype == 2) kMaximumScalar<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte)val, (ubyte*)target->data_device, len);
	if (mat->dtype == 3) kMaximumScalar<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig)val, (unsig*)target->data_device, len);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int max_by_axis(cudamat* mat, cudamat* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans)
        return ERROR_TRANSPOSED;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

		if (mat->dtype == 0) kMaxColumnwise<float><<<w,32>>>((float*)mat->data_device, (float*)target->data_device, w, h, std::numeric_limits<float>::min());
		if (mat->dtype == 1) kMaxColumnwise<doubl><<<w,32>>>((doubl*)mat->data_device, (doubl*)target->data_device, w, h, std::numeric_limits<doubl>::min());
		if (mat->dtype == 2) kMaxColumnwise<ubyte><<<w,32>>>((ubyte*)mat->data_device, (ubyte*)target->data_device, w, h, 0);
		if (mat->dtype == 3) kMaxColumnwise<unsig><<<w,32>>>((unsig*)mat->data_device, (unsig*)target->data_device, w, h, 0);

        CUDA_THREAD_SYNC();
    } else
        return ERROR_UNSUPPORTED;

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int min_by_axis(cudamat* mat, cudamat* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans)
        return ERROR_TRANSPOSED;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

		if (mat->dtype == 0) kMinColumnwise<float><<<w,32>>>((float*)mat->data_device, (float*)target->data_device, w, h, std::numeric_limits<float>::max());
		if (mat->dtype == 1) kMinColumnwise<doubl><<<w,32>>>((doubl*)mat->data_device, (doubl*)target->data_device, w, h, std::numeric_limits<doubl>::max());
		if (mat->dtype == 2) kMinColumnwise<ubyte><<<w,32>>>((ubyte*)mat->data_device, (ubyte*)target->data_device, w, h, 0);
		if (mat->dtype == 3) kMinColumnwise<unsig><<<w,32>>>((unsig*)mat->data_device, (unsig*)target->data_device, w, h, 0);

        CUDA_THREAD_SYNC();
    } else
        return ERROR_UNSUPPORTED;

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int sign(cudamat* mat, cudamat* target) {
    int len = mat->size[0]*mat->size[1];
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kSign<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, len);
	else if (mat->dtype == 1) kSign<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int apply_relu(cudamat* mat, cudamat* target, cudamat* dtarget) {
    unsigned int len = mat->size[0] * mat->size[1];

	if (!dtarget)
		return maximum_scalar(mat,0,target);

	if (!mat->on_device || !target->on_device || !dtarget->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->size[0] != dtarget->size[0] || mat->size[1] != dtarget->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype || mat->dtype != dtarget->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kApplyReluD<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, (float*)dtarget->data_device, len);
	else if (mat->dtype == 1) kApplyReluD<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, (doubl*)dtarget->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

	CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int apply_sigmoid(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kApplySigmoid<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, len);
	else if (mat->dtype == 1) kApplySigmoid<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int apply_sigmoid_deriv(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kApplySigmoidDeriv<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, len);
	else if (mat->dtype == 1) kApplySigmoidDeriv<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int apply_tanh(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kApplyTanh<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, len);
	else if (mat->dtype == 1) kApplyTanh<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int apply_tanh_deriv(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kApplyTanhDeriv<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, len);
	else if (mat->dtype == 1) kApplyTanhDeriv<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int apply_abs(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	if (mat->dtype == 0) kApplyAbs<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, len);
	if (mat->dtype == 1) kApplyAbs<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, len);
	/* do nothing for unsigned types */ 

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int apply_log_1_plus_exp(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kApplyLog1PlusExp<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, len);
	else if (mat->dtype == 1) kApplyLog1PlusExp<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int apply_log(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kLog<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, len);
	else if (mat->dtype == 1) kLog<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int apply_exp(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kExp<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, len);
	else if (mat->dtype == 1) kExp<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int apply_sqrt(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kSqrt<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, len);
	else if (mat->dtype == 1) kSqrt<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, len);
	else if (mat->dtype == 2) kSqrt<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int square(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kSquare<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, len);
	else if (mat->dtype == 1) kSquare<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, len);
	else if (mat->dtype == 2) kSquare<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int apply_pow(cudamat* mat, double pow, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kPow<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float)pow, (float*)target->data_device, len);
	else if (mat->dtype == 1) kPow<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl)pow, (doubl*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int apply_pow_matrix(cudamat* mat, cudamat* pow, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->size[0] != pow->size[0] || mat->size[1] != pow->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype || mat->dtype != pow->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kPowMatrix<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)pow->data_device, (float*)target->data_device, len);
	else if (mat->dtype == 1) kPowMatrix<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)pow->data_device, (doubl*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int reciprocal(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	     if (mat->dtype == 0) kReciprocal<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float*)target->data_device, len);
	else if (mat->dtype == 1) kReciprocal<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl*)target->data_device, len);
	else return ERROR_DTYPE_UNSUPPORTED;

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int dot(cudamat* mat1, cudamat* mat2, cudamat* target, double beta, double alpha) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
	if (mat1->dtype != mat2->dtype || mat1->dtype != target->dtype)
		return ERROR_MISMATCHED_DTYPE;
	if (mat1->dtype >= 2 || mat2->dtype >= 2 || target->dtype >= 2)
		return ERROR_DTYPE_UNSUPPORTED;

    if (get_leading_dimension(mat1) != get_leading_dimension(target) ||
        get_nonleading_dimension(mat2) != get_nonleading_dimension(target) ||
        get_nonleading_dimension(mat1) != get_leading_dimension(mat2)) {
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    }
    int m = get_leading_dimension(mat1),
        k = get_leading_dimension(mat2),
        n = get_nonleading_dimension(mat2);

	if (mat1->dtype == 0) {
		cublasSgemm(get_transpose_char(mat1), get_transpose_char(mat2), 
					m, n, k,
					(float)alpha, (const float*)mat1->data_device, mat1->size[0],
					(const float*)mat2->data_device, mat2->size[0],
					(float)beta, (float*)target->data_device, target->size[0]);
	} else {
		cublasDgemm(get_transpose_char(mat1), get_transpose_char(mat2), 
					m, n, k,
					(doubl)alpha, (const doubl*)mat1->data_device, mat1->size[0],
					(const doubl*)mat2->data_device, mat2->size[0],
					(doubl)beta, (doubl*)target->data_device, target->size[0]);
	}

    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}

DLLEXP extern double vdot(cudamat* mat1, cudamat* mat2, int* err_code) {
    int len = mat1->size[0]*mat1->size[1];
    double res;

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat1->is_trans != mat2->is_trans) {
        *err_code = ERROR_TRANSPOSEDNESS;
        return 0;
    }
    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1]) { 
        *err_code = ERROR_INCOMPATIBLE_DIMENSIONS;
        return 0;
    }
	if (mat1->dtype != mat2->dtype)
		return ERROR_MISMATCHED_DTYPE;
	if (mat1->dtype >= 2 || mat2->dtype >= 2)
		return ERROR_DTYPE_UNSUPPORTED;

	if (mat1->dtype == 0)  res = cublasSdot(len, (const float*)mat1->data_device, 1, (const float*)mat2->data_device, 1);
	if (mat1->dtype == 1)  res = cublasDdot(len, (const doubl*)mat1->data_device, 1, (const doubl*)mat2->data_device, 1);

    if (check_cublas_error()) {
        *err_code = CUBLAS_ERROR;
        return -1.;
    } else {
        *err_code = 0;
        return res;
    }
}

/* Perform the operation mat1 = mat1 + alpha * mat2. mat1 and mat2 must
   have the same transposedness. */
DLLEXP extern int add_mult(cudamat* mat1, cudamat* mat2, double alpha) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
	if (mat1->dtype != mat2->dtype)
		return ERROR_MISMATCHED_DTYPE;
	if (mat1->dtype >= 2 || mat2->dtype >= 2)
		return ERROR_DTYPE_UNSUPPORTED;

	if (mat1->dtype == 0)   cublasSaxpy(len, (float)alpha, (const float*)mat2->data_device, 1, (float*)mat1->data_device, 1);
	else                    cublasDaxpy(len, (doubl)alpha, (const doubl*)mat2->data_device, 1, (doubl*)mat1->data_device, 1);

    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}


DLLEXP extern int add_transpose(cudamat* mat1, cudamat* mat2, cudamat* target) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat1->size[0] != mat2->size[1] || mat1->size[1] != mat2->size[0] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat1->dtype != target->dtype || mat2->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    unsigned int height = mat2->size[0];
    unsigned int width  = mat2->size[1];

    // setup execution parameters
    unsigned int grid_x = height / COPY_BLOCK_SIZE;
    if (height % COPY_BLOCK_SIZE)
        grid_x++;

    unsigned int grid_y = width / COPY_BLOCK_SIZE;
    if (width % COPY_BLOCK_SIZE)
        grid_y++;

    dim3 grid(grid_x, grid_y, 1);
    dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);

	if (mat1->dtype == 0) kAddTrans<float><<<grid,threads>>>((float*)mat1->data_device, (float*)mat2->data_device, (float*)target->data_device, height,width);
	if (mat1->dtype == 1) kAddTrans<doubl><<<grid,threads>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (doubl*)target->data_device, height,width);
	if (mat1->dtype == 2) kAddTrans<ubyte><<<grid,threads>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (ubyte*)target->data_device, height,width);
	if (mat1->dtype == 3) kAddTrans<unsig><<<grid,threads>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (unsig*)target->data_device, height,width);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int add_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat1->dtype != target->dtype || mat2->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	if (mat1->dtype == 0) kAdd<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat1->data_device, (float*)mat2->data_device, (float*)target->data_device, len);
	if (mat1->dtype == 1) kAdd<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (doubl*)target->data_device, len);
	if (mat1->dtype == 2) kAdd<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (ubyte*)target->data_device, len);
	if (mat1->dtype == 3) kAdd<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (unsig*)target->data_device, len);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int subtract_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat1->dtype != target->dtype || mat2->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	if (mat1->dtype == 0) kSubtract<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat1->data_device, (float*)mat2->data_device, (float*)target->data_device, len);
	if (mat1->dtype == 1) kSubtract<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (doubl*)target->data_device, len);
	if (mat1->dtype == 2) kSubtract<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (ubyte*)target->data_device, len);
	if (mat1->dtype == 3) kSubtract<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (unsig*)target->data_device, len);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int subtract_transpose(cudamat* mat1, cudamat* mat2, cudamat* target) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat1->size[0] != mat2->size[1] || mat1->size[1] != mat2->size[0] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat1->dtype != target->dtype || mat2->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

    unsigned int height = mat2->size[0];
    unsigned int width  = mat2->size[1];

    // setup execution parameters
    unsigned int grid_x = height / COPY_BLOCK_SIZE;
    if (height % COPY_BLOCK_SIZE)
        grid_x++;

    unsigned int grid_y = width / COPY_BLOCK_SIZE;
    if (width % COPY_BLOCK_SIZE)
        grid_y++;

    dim3 grid(grid_x, grid_y, 1);
    dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);

	if (mat1->dtype == 0) kSubtractTrans<float><<<grid,threads>>>((float*)mat1->data_device, (float*)mat2->data_device, (float*)target->data_device, height,width);
	if (mat1->dtype == 1) kSubtractTrans<doubl><<<grid,threads>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (doubl*)target->data_device, height,width);
	if (mat1->dtype == 2) kSubtractTrans<ubyte><<<grid,threads>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (ubyte*)target->data_device, height,width);
	if (mat1->dtype == 3) kSubtractTrans<unsig><<<grid,threads>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (unsig*)target->data_device, height,width);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int divide_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat1->dtype != target->dtype || mat2->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	if (mat1->dtype == 0) kDivide<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat1->data_device, (float*)mat2->data_device, (float*)target->data_device, len);
	if (mat1->dtype == 1) kDivide<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (doubl*)target->data_device, len);
	if (mat1->dtype == 2) kDivide<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (ubyte*)target->data_device, len);
	if (mat1->dtype == 3) kDivide<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (unsig*)target->data_device, len);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

/* Elementwise multiplication of 2 matrices */
DLLEXP extern int mult_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat1->dtype != target->dtype || mat2->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	if (mat1->dtype == 0) kMult<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat1->data_device, (float*)mat2->data_device, (float*)target->data_device, len);
	if (mat1->dtype == 1) kMult<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat1->data_device, (doubl*)mat2->data_device, (doubl*)target->data_device, len);
	if (mat1->dtype == 2) kMult<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat1->data_device, (ubyte*)mat2->data_device, (ubyte*)target->data_device, len);
	if (mat1->dtype == 3) kMult<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat1->data_device, (unsig*)mat2->data_device, (unsig*)target->data_device, len);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int assign_array(cudamat* src, cudamat* dst) {
	int len = src->size[0]*src->size[1];

	if (!src->on_device || !dst->on_device)
		return ERROR_NOT_ON_DEVICE;
	if (src->size[0] != dst->size[0] || src->size[1] != dst->size[1])
		return ERROR_INCOMPATIBLE_DIMENSIONS;

	if (src->dtype == 0) {
		if (dst->dtype == 1) kAssignArray<float,doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)src->data_device, (doubl*)dst->data_device, len);
		if (dst->dtype == 2) kAssignArray<float,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)src->data_device, (ubyte*)dst->data_device, len);
		if (dst->dtype == 3) kAssignArray<float,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)src->data_device, (unsig*)dst->data_device, len);
	} else if (src->dtype == 1) {
		if (dst->dtype == 0) kAssignArray<doubl,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)src->data_device, (float*)dst->data_device, len);
		if (dst->dtype == 2) kAssignArray<doubl,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)src->data_device, (ubyte*)dst->data_device, len);
		if (dst->dtype == 3) kAssignArray<doubl,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)src->data_device, (unsig*)dst->data_device, len);
	} else if (src->dtype == 2) {
		if (dst->dtype == 0) kAssignArray<ubyte,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)src->data_device, (float*)dst->data_device, len);
		if (dst->dtype == 1) kAssignArray<ubyte,doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)src->data_device, (doubl*)dst->data_device, len);
		if (dst->dtype == 3) kAssignArray<ubyte,unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)src->data_device, (unsig*)dst->data_device, len);
	} else if (src->dtype == 3) {
		if (dst->dtype == 0) kAssignArray<unsig,float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)src->data_device, (float*)dst->data_device, len);
		if (dst->dtype == 1) kAssignArray<unsig,doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)src->data_device, (doubl*)dst->data_device, len);
		if (dst->dtype == 2) kAssignArray<unsig,ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)src->data_device, (ubyte*)dst->data_device, len);
	}

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int assign_scalar(cudamat* mat, double alpha) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

	if (mat->dtype == 0) kAssignScalar<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float)alpha, len);
	if (mat->dtype == 1) kAssignScalar<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl)alpha, len);
	if (mat->dtype == 2) kAssignScalar<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte)alpha, len);
	if (mat->dtype == 3) kAssignScalar<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig)alpha, len);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int mult_by_scalar(cudamat* mat, double alpha, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	if (mat->dtype == 0) kMultScalar<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float)alpha, (float*)target->data_device, len);
	if (mat->dtype == 1) kMultScalar<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl)alpha, (doubl*)target->data_device, len);
	if (mat->dtype == 2) kMultScalar<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte)alpha, (ubyte*)target->data_device, len);
	if (mat->dtype == 3) kMultScalar<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig)alpha, (unsig*)target->data_device, len);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int divide_by_scalar(cudamat* mat, double alpha, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	if (mat->dtype == 0) kDivideScalar<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float)alpha, (float*)target->data_device, len);
	if (mat->dtype == 1) kDivideScalar<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl)alpha, (doubl*)target->data_device, len);
	if (mat->dtype == 2) kDivideScalar<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte)alpha, (ubyte*)target->data_device, len);
	if (mat->dtype == 3) kDivideScalar<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig)alpha, (unsig*)target->data_device, len);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern int add_scalar(cudamat* mat, double alpha, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;

	if (mat->dtype == 0) kAddScalar<float><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((float*)mat->data_device, (float)alpha, (float*)target->data_device, len);
	if (mat->dtype == 1) kAddScalar<doubl><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((doubl*)mat->data_device, (doubl)alpha, (doubl*)target->data_device, len);
	if (mat->dtype == 2) kAddScalar<ubyte><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((ubyte*)mat->data_device, (ubyte)alpha, (ubyte*)target->data_device, len);
	if (mat->dtype == 3) kAddScalar<unsig><<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>((unsig*)mat->data_device, (unsig)alpha, (unsig*)target->data_device, len);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

DLLEXP extern double euclid_norm(cudamat* mat, int* err_code) {
    int len = mat->size[0]*mat->size[1];
    if (mat->dtype >= 2)
        return ERROR_DTYPE_UNSUPPORTED;

	double res;
	if (mat->dtype == 0) res = cublasSnrm2(len, (const float*)mat->data_device, 1);
	else                 res = cublasDnrm2(len, (const doubl*)mat->data_device, 1);

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (check_cublas_error()) {
        *err_code = CUBLAS_ERROR;
        return -1.;
    } else {
        *err_code = 0;
        return res;
    }
}

DLLEXP extern int selectRows(cudamat* source, cudamat* target, cudamat* indices){
    const int nRetRows = indices->size[1];

    if (source->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;
    if (indices->dtype != 0)
        return ERROR_DTYPE_UNSUPPORTED;

    if (nRetRows==0) return 0;

    dim3 gridDim((nRetRows+31)/32);
    dim3 blockDim(32);

	// TODO: support integer indices
	if (source->dtype == 0) kSelectRows<float><<<gridDim, blockDim>>>((float*)source->data_device, (float*)target->data_device, (float*)indices->data_device, nRetRows, source->size[0], source->size[1]);
	if (source->dtype == 1) kSelectRows<doubl><<<gridDim, blockDim>>>((doubl*)source->data_device, (doubl*)target->data_device, (float*)indices->data_device, nRetRows, source->size[0], source->size[1]);
	if (source->dtype == 2) kSelectRows<ubyte><<<gridDim, blockDim>>>((ubyte*)source->data_device, (ubyte*)target->data_device, (float*)indices->data_device, nRetRows, source->size[0], source->size[1]);
	if (source->dtype == 3) kSelectRows<unsig><<<gridDim, blockDim>>>((unsig*)source->data_device, (unsig*)target->data_device, (float*)indices->data_device, nRetRows, source->size[0], source->size[1]);

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

DLLEXP extern int setSelectedRows(cudamat* target, cudamat* source, cudamat* indices){
    const int nSetRows = indices->size[1];

    if (source->dtype != target->dtype)
        return ERROR_MISMATCHED_DTYPE;
    if (indices->dtype != 0)
        return ERROR_DTYPE_UNSUPPORTED;

    if (nSetRows==0)
        return 0;

    dim3 gridDim((nSetRows+31)/32);
    dim3 blockDim(32);

    if (source->dtype == 0) kSetSelectedRows<float><<<gridDim, blockDim>>>((float*)target->data_device, (float*)source->data_device, (float*)indices->data_device, nSetRows, target->size[0], target->size[1]);
    if (source->dtype == 1) kSetSelectedRows<doubl><<<gridDim, blockDim>>>((doubl*)target->data_device, (doubl*)source->data_device, (float*)indices->data_device, nSetRows, target->size[0], target->size[1]);
    if (source->dtype == 2) kSetSelectedRows<ubyte><<<gridDim, blockDim>>>((ubyte*)target->data_device, (ubyte*)source->data_device, (float*)indices->data_device, nSetRows, target->size[0], target->size[1]);
    if (source->dtype == 3) kSetSelectedRows<unsig><<<gridDim, blockDim>>>((unsig*)target->data_device, (unsig*)source->data_device, (float*)indices->data_device, nSetRows, target->size[0], target->size[1]);
    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

DLLEXP extern int dropout(rnd_struct* rnd_state, cudamat* matA, cudamat* matB, float rate,
                          cudamat* targetA, cudamat* targetB) {
    unsigned int len = matA->size[0] * matA->size[1];

    if (!matA->on_device || !targetA->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (matA->size[0] != targetA->size[0] || matA->size[1] != targetA->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (matA->dtype != targetA->dtype)
        return ERROR_MISMATCHED_DTYPE;

    
	if (matB) {
		if (!matB->on_device || !targetB->on_device)
			return ERROR_NOT_ON_DEVICE;
		if (matB->dtype != targetB->dtype)
			return ERROR_MISMATCHED_DTYPE;
		if (matB->size[0] != targetB->size[0] || matB->size[1] != targetB->size[1])
			return ERROR_INCOMPATIBLE_DIMENSIONS;
		if (matA->size[0] != matB->size[0] || matA->size[1] != matB->size[1])
			return ERROR_INCOMPATIBLE_DIMENSIONS;
		if (matB->dtype >= 2)
			return ERROR_DTYPE_UNSUPPORTED;
	}

	if (matA->dtype == 0) {
		if (matB && matB->dtype != 0)
			return ERROR_DTYPE_UNSUPPORTED;
		kDropout<float,float><<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, 
					                                                        (float*)matA->data_device,    (float*)(matB ? matB : matA)->data_device, rate,
					                                                        (float*)targetA->data_device, (float*)(targetB ? targetB : targetA)->data_device, len);
	} else if (matA->dtype == 1) {
		if (matB && matB->dtype != 1)
			return ERROR_DTYPE_UNSUPPORTED;
		kDropout<doubl,doubl><<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, 
					                                                        (doubl*)matA->data_device,    (doubl*)(matB ? matB : matA)->data_device, rate,
					                                                        (doubl*)targetA->data_device, (doubl*)(targetB ? targetB : targetA)->data_device, len);
	} else if (matA->dtype == 2) {
		if (!matB)
			return ERROR_DTYPE_UNSUPPORTED;
		if (matB->dtype == 0)
			kDropout<ubyte,float><<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, 
																				(ubyte*)matA->data_device,    (float*)(matB ? matB : matA)->data_device, rate,
																				(ubyte*)targetA->data_device, (float*)(targetB ? targetB : targetA)->data_device, len);
		if (matB->dtype == 1)
			kDropout<ubyte,doubl><<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, 
																				(ubyte*)matA->data_device,    (doubl*)(matB ? matB : matA)->data_device, rate,
																				(ubyte*)targetA->data_device, (doubl*)(targetB ? targetB : targetA)->data_device, len);
	}

    CUDA_THREAD_SYNC();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

}
