CUDA_LIB :=/usr/local/cuda/lib64 -lcuda -lcudart -lcublas

CUBLAS_LIB :=/usr/local/cuda/include

all: CPU GPU
	g++ cpu_main.o cuda_main.o -o test.run -L $(CUDA_LIB)
CPU:
	g++ cpu_main.cpp -c -I $(CUBLAS_LIB)
GPU:
	nvcc -arch=sm_61 cuda_main.cu -c -I $(CUBLAS_LIB)
clean:
	rm *.o *.run *.txt
