CUDA_LIB :=/usr/local/cuda/lib64 -lcuda -lcudart -lcusparse_static -lculibos

CUSPARSE_LIB :=/usr/local/cuda/include

all: CPU GPU
	g++ cpu_main.o cuda_main.o -o test.run -L $(CUDA_LIB) -lpthread -ldl
CPU:
	g++ cpu_main.cpp -c -I $(CUSPARSE_LIB)
GPU:
	nvcc -arch=sm_35 cuda_main.cu -c -I $(CUSPARSE_LIB)
clean:
	rm *.o *.run
