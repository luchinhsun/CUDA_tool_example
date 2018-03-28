CUDA_PATH :=/usr/local/cuda
CUDA_LIB := -L $(CUDA_PATH)/lib64 -lcuda -lcudart
CUDA_INCLUDES := -I $(CUDA_PATH)/include

All: cpu gpu cmp

cmp:
	mpiCC $(CUDA_LIB) $(CUDA_INCLUDES) cudamain.o MPI.o
cpu:
	mpiCC MPI.c -c
gpu:
	nvcc cudamain.cu -c
run:
	qsub pbs.sh
check:
	qstat
err:
	cat *.err
clean:
	rm *.err *.log *.txt *.o a.out
