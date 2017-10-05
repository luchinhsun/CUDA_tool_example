#include "head.h"

//variable for cusparse
cusparseStatus_t status;
cusparseHandle_t handle=0;

float * yHostPtr;
float * y;
float * xHostPtr;
float * x;
int * xHostInd;
int * xInd;
float * result;
float * Hostresult;

void Allocate_Memory_and_Init(){
	//cusparse
	size_t size = nnz*sizeof(int);

	Hostresult    = (float *)malloc(1*sizeof(float));
	yHostPtr    = (float *)malloc(n*sizeof(float));
	yHostPtr[0] = 50.0; yHostPtr[1] = 60.0; yHostPtr[2] = 70.0; yHostPtr[3] = 80.0;
	xHostPtr    = (float *)malloc(nnz*sizeof(float));
	xHostPtr[0] = 20.0; xHostPtr[1] = 11.0;// xHostPtr[2] = 0.0; xHostPtr[3] = 0.0;
	xHostInd = (int *) malloc(size);
	xHostInd[0] = 1.0; xHostInd[1] = 3.0;

	cudaError_t Error;

	Error = cudaMalloc((void**)&y, n*sizeof(float));
	printf("CUDA error(malloc y) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&x, nnz*sizeof(float));
        printf("CUDA error(malloc x) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&xInd, nnz*sizeof(int));
        printf("CUDA error(malloc xInd) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&result, 1*sizeof(float));
        printf("CUDA error(malloc result) = %s\n",cudaGetErrorString(Error));

	status= cusparseCreate(&handle);
}

void Send_To_Device(){
	cudaError_t Error;
	size_t size = nnz*sizeof(int);

	Error = cudaMemcpy(xInd, xHostInd, size, cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy xInd) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(y, yHostPtr, (size_t)(n*sizeof(float)), cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy y) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(x, xHostPtr, (size_t)(n*sizeof(float)), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy x) = %s\n",cudaGetErrorString(Error));
}

void Call_GPUFunction(){
	cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
	cusparseSdoti(handle, nnz, x, xInd, y, result, CUSPARSE_INDEX_BASE_ZERO);

}


void Send_To_Host(){
	cudaError_t Error;

	Error = cudaMemcpy(Hostresult, result, (size_t)(1*sizeof(float)), cudaMemcpyDeviceToHost);
        printf("CUDA error(memcpy x->xHostPtr) = %s\n",cudaGetErrorString(Error));
}

void Free_Memory(){
	if (yHostPtr) free(yHostPtr);
	if (xHostPtr) free(xHostPtr);
        if (y) cudaFree(y);
	if (x) cudaFree(x);

        if (handle) cusparseDestroy(handle);

	if (Hostresult) free(Hostresult);
	if (result) cudaFree(result);
	if (xHostInd) free(xHostInd);
	if (xInd) cudaFree(xInd);
}
