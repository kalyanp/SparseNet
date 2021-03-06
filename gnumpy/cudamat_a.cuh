#define ERROR_INCOMPATIBLE_DIMENSIONS -1
#define CUBLAS_ERROR -2
#define CUDA_ERROR -3
#define VIEW_ERROR -4
#define ERROR_TRANSPOSED -5
#define ERROR_GENERIC -6
#define ERROR_TRANSPOSEDNESS -7
#define ERROR_NOT_ON_DEVICE -8
#define ERROR_UNSUPPORTED -9
#define ERROR_MISMATCHED_DTYPE -10
#define ERROR_DTYPE_UNSUPPORTED -11

struct cudamat {
    void* data_host;
    void* data_device;
    int on_device;
    int on_host;
    int size[2];
    int is_trans; // 0 or 1
    int owns_data;
	int dtype;   // 0 = float, 1 = unsigned char
};

struct rnd_struct {
    unsigned int* dev_mults;
    unsigned long long* dev_words;
};

