#include "cudamat_kernels.cuh"
#include "float.h"

typedef unsigned char uchar;
typedef unsigned int  uint;

template <typename T> struct select4 { };
template <> struct select4<float>  { typedef float4  type; template <typename A> __device__ inline static float4  make(const A& a) { return make_float4 (a.x,a.y,a.z,a.w); }};
template <> struct select4<double> { typedef double4 type; template <typename A> __device__ inline static double4 make(const A& a) { return make_double4(a.x,a.y,a.z,a.w); }};
template <typename R> inline __device__ typename select4<R>::type saturate(float4 a) { return select4<R>::make(make_float4(__saturatef(a.x),__saturatef(a.y),__saturatef(a.z),__saturatef(a.w))); }


#define DEFINE_FUNCTIONS(T) \
	template <typename A> inline __device__ __host__ T##4 make_##T##4(A a) { return make_##T##4((T)a.x,(T)a.y,(T)a.z,(T)a.w); } \
	DEFINE_OPERATORS(T,+) \
	DEFINE_OPERATORS(T,-) \
	DEFINE_OPERATORS(T,*) \
	DEFINE_OPERATORS(T,/)

#define DEFINE_OPERATORS(T,OP) \
	inline __device__ __host__ T##4 operator OP (T##4 a, T##4 b)  { return make_##T##4(a.x OP b.x, a.y OP b.y, a.z OP b.z, a.w OP b.w); } \
	inline __device__ __host__ T##4 operator OP (T    a, T##4 b)  { return make_##T##4(a   OP b.x, a   OP b.y, a   OP b.z, a   OP b.w); } \
	inline __device__ __host__ T##4 operator OP (T##4 a, T    b)  { return make_##T##4(a.x OP b  , a.y OP b  , a.z OP b  , a.w OP b  ); } \
	inline __device__ __host__ T##4 operator OP=(T##4& a, T##4 b) { a.x OP= b.x; a.y OP= b.y; a.z OP= b.z; a.w OP= b.w; return a; } \
	inline __device__ __host__ T##4 operator OP=(T##4& a, T    b) { a.x OP= b;   a.y OP= b;   a.z OP= b;   a.w OP= b;   return a; } 

//DEFINE_FUNCTIONS(uchar)
//DEFINE_FUNCTIONS(uint)
//DEFINE_FUNCTIONS(float)
//DEFINE_FUNCTIONS(double)





/* ------------------------- Random number generation ------------------------- */

__device__ __forceinline__ float  _exp(float   x) { return __expf(x); }
__device__ __forceinline__ double _exp(double  x) { return   exp(x);  }
__device__ __forceinline__ float  _log(float   x) { return __logf(x); }
__device__ __forceinline__ double _log(double  x) { return   log(x);  }
__device__ __forceinline__ float  _sqrt(float  x) { return sqrtf(x);  }
__device__ __forceinline__ double _sqrt(double x) { return  sqrt(x);  }
__device__ __forceinline__ float  _pow(float  x, float p)  { return __powf(x,p);  }
__device__ __forceinline__ double _pow(double x, double p) { return   pow(x,p);  }


__device__ __forceinline__ unsigned char _exp(unsigned char x)  { return  (unsigned char)_exp((float)x);  }
__device__ __forceinline__ unsigned char _log(unsigned char x)  { return  (unsigned char)_log((float)x);  }
__device__ __forceinline__ unsigned char _sqrt(unsigned char x) { return  (unsigned char)_sqrt((float)x);  }

__device__ __forceinline__ unsigned  _exp(unsigned x)  { return  (unsigned )_exp((double)x);  }
__device__ __forceinline__ unsigned  _log(unsigned x)  { return  (unsigned )_log((double)x);  }
__device__ __forceinline__ unsigned  _sqrt(unsigned x) { return  (unsigned )_sqrt((double)x);  }

__global__ void kSeedRandom(unsigned int* rndMults, unsigned long long* rndWords, unsigned int seed) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // The initial x is the seed and the initial carry is 1
    unsigned long long rndWord = ((unsigned long long)seed << 32) + 1;
    const unsigned int rndMult = rndMults[idx];
    /*
     * Run the chain for a few steps so that all the streams have a chance
     * to differentiate. They start out generating similar random numbers
     * because all the multipliers are similar.
     */
    #pragma unroll
    for(unsigned int i = 0; i < NUM_RND_BURNIN; i++) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    }
    rndWords[idx] = rndWord;
}

__global__ void kRandomUniform(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    #pragma unroll
    for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        gData[i] = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
    }
    rndWords[idx] = rndWord;
}

__global__ void kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    float rnd1, rnd2, R, T;
    #pragma unroll
    for(unsigned int i = idx; i < numElements; i += 2*NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd1 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd2 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        T = 2 * PI * rnd2;
        R = sqrtf(-2 * __logf(rnd1));
        gData[i] = R * __cosf(T);
        if (i + NUM_RND_STREAMS < numElements)
            gData[i + NUM_RND_STREAMS] = R * __sinf(T);
    }
    rndWords[idx] = rndWord;
}

/* ------------------------- Data copying ------------------------- */

/*
Copy row slice from source to target. There is a block for every 32x32 chunk being copied.
*/
template <typename T>
__global__ void kGetRowSlice(T* source, T* target, int start, int end, int width, int height) {
    const int row = start + blockIdx.x * 32 + threadIdx.x;
    const int start_col = blockIdx.y * 32;

    const int end_col = (start_col + 32 < width) ? start_col + 32: width;

    const int target_height = end - start;

    if (row < end) {
        for (int cur_col = start_col; cur_col < end_col; cur_col++)
            target[cur_col * target_height + row - start] = source[cur_col * height + row];
    }
}

template <typename T>
__global__ void kSetRowSlice(T* source, T* target, int start, int end, int width, int height) {
    const int row = start + blockIdx.x * 32 + threadIdx.x;
    const int start_col = blockIdx.y * 32;

    const int end_col = (start_col + 32 < width) ? start_col + 32: width;

    const int source_height = end - start;

    if (row < end) {
        #pragma unroll
        for (int cur_col = start_col; cur_col < end_col; cur_col++)
            target[cur_col * height + row] = source[cur_col * source_height + row - start];
            //source[cur_col * height + row - start] = target[cur_col * target_height + row];
    }
}

template <typename T> 
__global__ void kTranspose(T *odata, T *idata, int width, int height) {
    __shared__ T block[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < width) && (yIndex < height)) {
        unsigned int index_in = yIndex * width + xIndex;

        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < height) && (yIndex < width)) {
        unsigned int index_out = yIndex * height + xIndex;

        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

/* ------------------------- Mathematical operations ------------------------- */

template <typename T, typename R>
__global__ void kLessThan(T* mat1, T* mat2, R* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = (R)(mat1[i] < mat2[i]);
    }
}

template <typename T, typename R>
__global__ void kLessThanScalar(T* mat, T val, R* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = (R)(mat[i] < val);
    }
}

template <typename T, typename R>
__global__ void kGreaterThan(T* mat1, T* mat2, R* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = (R)(mat1[i] > mat2[i]);
    }
}

template <typename T, typename R>
__global__ void kGreaterThanScalar(T* mat, T val, R* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = (R)(mat[i] > val);
    }
}

template <typename T>
__global__ void kMaximum(T* mat1, T* mat2, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = max(mat1[i],mat2[i]);
    }
}

template <typename T>
__global__ void kMaximumScalar(T* mat, T val, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = max(mat[i],val);
    }
}

template <typename T>
__global__ void kMaxColumnwise(T* mat, T* target, unsigned int width, unsigned int height, T type_min) {
    __shared__ T max_vals[32];
    T cur_max = type_min;
    T val = 0;
 
    #pragma unroll
    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[blockIdx.x * height + i];

        if (val > cur_max)
            cur_max = val;
    }

    max_vals[threadIdx.x] = cur_max;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = type_min;

        #pragma unroll
        for (unsigned int i = 0; i < 32; i++)
            if (max_vals[i] > cur_max)
                cur_max = max_vals[i];

        target[blockIdx.x] = cur_max;
    }
}

template <typename T>
__global__ void kMinColumnwise(T* mat, T* target, unsigned int width, unsigned int height, T type_max) {
    __shared__ T min_vals[32];
    T cur_min = type_max;
    T val = 0;
 
    #pragma unroll
    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[blockIdx.x * height + i];

        if (val < cur_min)
            cur_min = val;
    }

    min_vals[threadIdx.x] = cur_min;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_min = type_max;

        #pragma unroll
        for (unsigned int i = 0; i < 32; i++)
            if (min_vals[i] < cur_min)
                cur_min = min_vals[i];

        target[blockIdx.x] = cur_min;
    }
}

template <typename T>
__global__ void kSign(T* mat, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i] ? copysignf(1., mat[i]) : 0.;
    }
}

template <typename T>
__global__ void kApplyReluD(T* mat, T* target, T* dtarget, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
		T test = mat[i] > 0 ? 1 : 0;
		target[i] = mat[i]*test;
		dtarget[i] = 1*test;
    }
}

template <typename T>
__global__ void kApplySigmoid(T* mat, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = 1 / (1 + _exp(-mat[i]));
    }
}

template <typename T>
__global__ void kApplySigmoidDeriv(T* mat, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i]*(1-mat[i]);
    }
}

template <typename T>
__global__ void kApplyTanh(T* mat, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    T mat_i, exp2x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        mat_i = mat[i];
        exp2x = _exp(2 * mat_i);
        target[i] = 1 - 2 / (exp2x + 1);
    }
}

template <typename T>
__global__ void kApplyTanhDeriv(T* mat, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = 1-mat[i]*mat[i];
    }
}

template <typename T>
__global__ void kApplyAbs(T* mat, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    
    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i] * ((mat[i] > 0) - (mat[i] < 0));
    }
}

template <typename T>
__global__ void kApplyLog1PlusExp(T* mat, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    T mat_i;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        mat_i = mat[i];
        if (mat_i > 0)
            target[i] = (_log(1 + _exp(-mat_i)) + mat_i);
        else
            target[i] = _log(1 + _exp(mat_i));
    }
}

template <typename T>
__global__ void kLog(T* mat, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = _log(mat[i]);
    }
}

template <typename T>
__global__ void kExp(T* mat, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = _exp(mat[i]);
    }
}

template <typename T>
__global__ void kSqrt(T* mat, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = _sqrt(mat[i]);
    }
}

template <typename T>
__global__ void kSquare(T* mat, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i]*mat[i];
    }
}

template <typename T>
__global__ void kPow(T* mat, T exponent, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = _pow(mat[i], exponent);
    }
}

template <typename T>
__global__ void kPowMatrix(T* mat, T* exponent, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = _pow(mat[i], exponent[i]);
    }
}

template <typename T>
__global__ void kReciprocal(T* mat, T* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads)
        target[i] = (T)1 / mat[i];
}

template <typename T>
__global__ void kDiffCols(T* mat, T* target, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < (width-1) * height; i += numThreads) {
        target[i] = mat[i+height] - mat[i];
    }
}

template <typename T>
__global__ void kAddColVector(T* mat, T* vec, T* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] + vec[i % height];
    }
}

template <typename T>
__global__ void kSubColVector(T* mat, T* vec, T* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] - vec[i % height];
    }
}

template <typename T>
__global__ void kAddRowVector(T* mat, T* vec, T* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] + vec[i / height];
    }
}

template <typename T>
__global__ void kSubRowVector(T* mat, T* vec, T* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] - vec[i / height];
    }
}

template <typename T>
__global__ void kAddColMult(T* mat, T* vec, T* tgtMat, T mult, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] + mult * vec[i % height];
    }
}

template <typename T>
__global__ void kMultByColVector(T* mat, T* vec, T* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] * vec[i % height];
    }
}

template <typename T>
__global__ void kMultByRowVector(T* mat, T* vec, T* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] * vec[i / height];
    }
}

template <typename T>
__global__ void kClipNorm(T* mat, T* vec, T maxnorm_sqr, T* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < width * height; i += numThreads) {
        T val = vec[i % height];
		T scale = val > maxnorm_sqr ? (maxnorm_sqr/_sqrt(val)) : 1;
        tgtMat[i] = scale*mat[i];
    }
}

template <typename T>
__global__ void kAdd(T* a, T* b, T* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] + b[i];
    }
}

template <typename T>
__global__ void kAddTrans(T* a, T* b, T* dest, unsigned int width, unsigned int height) {
    __shared__ T block[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < width) && (yIndex < height)) {
        unsigned int index_in = yIndex * width + xIndex;

        block[threadIdx.y][threadIdx.x] = b[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < height) && (yIndex < width)) {
        unsigned int index_out = yIndex * height + xIndex;

        dest[index_out] = a[index_out] + block[threadIdx.x][threadIdx.y];
    }
}

template <typename T>
__global__ void kSubtract(T* a, T* b, T* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] - b[i];
    }
}

template <typename T>
__global__ void kSubtractTrans(T* a, T* b, T* dest, unsigned int width, unsigned int height) {
    __shared__ T block[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < width) && (yIndex < height)) {
        unsigned int index_in = yIndex * width + xIndex;

        block[threadIdx.y][threadIdx.x] = b[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < height) && (yIndex < width)) {
        unsigned int index_out = yIndex * height + xIndex;

        dest[index_out] = a[index_out] - block[threadIdx.x][threadIdx.y];
    }
}

template <typename T>
__global__ void kDivide(T* a, T* b, T* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] / b[i];
    }
}

template <typename T>
__global__ void kMult(T* a, T* b, T* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] * b[i];
    }
}

template <typename T>
__global__ void kMultScalar(T* mat, T alpha, T* dest, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        dest[i] = alpha * mat[i];
    }
}

template <typename T1,typename T2>
__global__ void kAssignArray(T1* src, T2* dst, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        dst[i] = (T2)src[i];
    }
}

template <typename T>
__global__ void kAssignScalar(T* dest, T alpha, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        dest[i] = alpha;
    }
}

template <typename T>
__global__ void kDivideScalar(T* mat, T alpha, T* dest, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < len; i += numThreads) {
        dest[i] = mat[i] / alpha;
    }
}

template <typename T>
__global__ void kAddScalar(T* a, T alpha, T* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] + alpha;
    }
}

template <typename T>
__global__ void kSelectRows(T* source, T* target, float* indices, int nRowIs, int nCols, int nSourceRows){
    __shared__ int sourceRowIndices[32];
    const int startTargetRowI = blockIdx.x * 32;
    const int tid = threadIdx.x;
    const int localNRowIs = min(32, nRowIs-startTargetRowI);

    // cooperatively load 32 row indices
    if (tid < localNRowIs){
        sourceRowIndices[tid] = int(indices[startTargetRowI + tid]);
        if (sourceRowIndices[tid]<0)
            sourceRowIndices[tid] += nSourceRows;
        if (sourceRowIndices[tid]<0 || sourceRowIndices[tid]>=nSourceRows)
            sourceRowIndices[tid] = -1;
    }
    __syncthreads();

    // copy 32 rows
    #pragma unroll
    for (int i=0; i<localNRowIs; i++){
        const int targetRowI = startTargetRowI + i, sourceRowI = sourceRowIndices[i];
		#pragma unroll
        for (int colI=tid; colI<nCols; colI+=32)
            target[targetRowI * nCols + colI] = sourceRowI==-1 ? (1.0/0.0 -1.0/0.0) : source[sourceRowI * nCols + colI];
    }
}

template <typename T>
__global__ void kSetSelectedRows(T* target, T* source, float* indices, int nRowIs, int nCols, int nTargetRows){
    __shared__ int targetRowIndices[32];
    const int startSourceRowI = blockIdx.x * 32;
    const int tid = threadIdx.x;
    const int localNRowIs = min(32, nRowIs-startSourceRowI);

    // cooperatively load 32 row indices
    if (tid < localNRowIs){
        targetRowIndices[tid] = int(indices[startSourceRowI + tid]);
        if (targetRowIndices[tid]<0)
            targetRowIndices[tid] += nTargetRows;
        if (targetRowIndices[tid]<0 || targetRowIndices[tid]>=nTargetRows)
            targetRowIndices[tid] = -1;
    }
    __syncthreads();

    // copy 32 rows
    #pragma unroll
    for (int i=0; i<localNRowIs; i++){
        const int sourceRowI = startSourceRowI + i, targetRowI = targetRowIndices[i];
	    #pragma unroll
        for (int colI=tid; colI<nCols; colI+=32)
            target[targetRowI * nCols + colI] = targetRowI==-1 ? (1.0/0.0 -1.0/0.0) : source[sourceRowI * nCols + colI];
    }
}

template <typename TA, typename TB>
__global__ void kDropout(unsigned int* rndMults, unsigned long long* rndWords, TA* matA, TB* matB, float rate, TA* targetA, TB* targetB, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    #pragma unroll
    for(unsigned int i = idx; i < len; i += NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        float trial = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        if (trial < rate) {
            targetB[i] = (TA)0;
            targetA[i] = (TB)0;
        } else {
            targetB[i] = matB[i];
            targetA[i] = matA[i];
        }
    }
    rndWords[idx] = rndWord;
}

