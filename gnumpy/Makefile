all: libcudamat.so libcudamat_conv.so

libcudamat.so:
	nvcc -O -Xlinker -rpath,/usr/local/cuda/lib --ptxas-options=-v --compiler-options '-fPIC' -o libcudamat.so --shared cudamat.cu -lcublas -lcuda -lcudart

libcudamat_conv.so: cudamat_conv.cu cudamat_conv_kernels.cu cudamat_conv.cuh cudamat.cuh cudamat_conv_kernels.cuh
	nvcc -O3 \
		-v \
		-gencode=arch=compute_30,code=sm_30 \
		-gencode=arch=compute_35,code=sm_35 \
		--compiler-options '-fPIC' -o libcudamat_conv.so \
		--shared cudamat_conv.cu cudamat_conv_kernels.cu -lcublas -L/usr/local/cuda/lib64
		
clean:
	@rm *.linkinfo *.pyc *.so
