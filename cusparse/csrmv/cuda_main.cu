#include "head.h"

//variable for cusparse
cusparseStatus_t status;
cusparseHandle_t handle=0;
cusparseMatDescr_t descr=0;
cusparseMatDescr_t descrL=0;
cusparseMatDescr_t descrU=0;
int *cooRowIndexHostPtr;
int * cooColIndexHostPtr;
float * cooValHostPtr;
int *cooRowIndex;
int * cooColIndex;
float * cooVal;
float * cooValLU;
float * yHostPtr;
float * y;
float * xHostPtr;
float * x;
float * temp;
int * csrRowPtr;

float dzero =0.0;
float done =1.0;
float dtwo =2.0;
float dthree=3.0;
float dfive =5.0;

void Allocate_Memory_and_Init(){
	//cusparse_Init
	int i;
	size_t size = nnz*sizeof(int);
	cooRowIndexHostPtr = (int *) malloc(size);
	cooColIndexHostPtr = (int *) malloc(size);
	cooValHostPtr = (float *)malloc(nnz*sizeof(float));
	cooRowIndexHostPtr[0]=0; cooColIndexHostPtr[0]=0; cooValHostPtr[0]=1.0;
	cooRowIndexHostPtr[1]=0; cooColIndexHostPtr[1]=2; cooValHostPtr[1]=2.0;
	cooRowIndexHostPtr[2]=0; cooColIndexHostPtr[2]=3; cooValHostPtr[2]=3.0;
	cooRowIndexHostPtr[3]=1; cooColIndexHostPtr[3]=1; cooValHostPtr[3]=4.0;
	cooRowIndexHostPtr[4]=2; cooColIndexHostPtr[4]=0; cooValHostPtr[4]=5.0;
	cooRowIndexHostPtr[5]=2; cooColIndexHostPtr[5]=2; cooValHostPtr[5]=6.0;
	cooRowIndexHostPtr[6]=2; cooColIndexHostPtr[6]=3; cooValHostPtr[6]=7.0;
	cooRowIndexHostPtr[7]=3; cooColIndexHostPtr[7]=1; cooValHostPtr[7]=8.0;
	cooRowIndexHostPtr[8]=3; cooColIndexHostPtr[8]=3; cooValHostPtr[8]=9.0;
	cooRowIndexHostPtr[9]=4; cooColIndexHostPtr[9]=0; cooValHostPtr[9]=10.0;
        cooRowIndexHostPtr[10]=4; cooColIndexHostPtr[10]=3; cooValHostPtr[10]=11.0;

	yHostPtr    = (float *)malloc(n*sizeof(float));
	yHostPtr[0] = 50.0; yHostPtr[1] = 60.0; yHostPtr[2] = 70.0; yHostPtr[3] = 80.0;
	xHostPtr    = (float *)malloc(m*sizeof(float));
	for(i=0;i<m;i++){
		xHostPtr[i] = 0.0;
	}

	cudaError_t Error;

	Error = cudaMalloc((void**)&cooRowIndex, size);
	printf("CUDA error(malloc RowIndex) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&cooColIndex, size);
	printf("CUDA error(malloc ColIndex) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&cooVal, nnz*sizeof(float));
	printf("CUDA error(malloc Val) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&cooValLU, nnz*sizeof(float));
        printf("CUDA error(malloc Val) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&y, n*sizeof(float));
	printf("CUDA error(malloc y) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&x, m*sizeof(float));
        printf("CUDA error(malloc x) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&csrRowPtr,(m+1)*sizeof(int));
        printf("CUDA error(malloc csrRowPtr) = %s\n",cudaGetErrorString(Error));

	status= cusparseCreate(&handle);
	status= cusparseCreateMatDescr(&descr);

	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
}

void Send_To_Device(){
	cudaError_t Error;
	size_t size = nnz*sizeof(int);
	Error = cudaMemcpy(cooRowIndex, cooRowIndexHostPtr, size, cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy RowIndex) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(cooColIndex, cooColIndexHostPtr, size, cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy ColIndex) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(cooVal, cooValHostPtr, (size_t)(nnz*sizeof(float)), cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy Val) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(y, yHostPtr, (size_t)(n*sizeof(float)), cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy y) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(x, xHostPtr, (size_t)(m*sizeof(float)), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy x) = %s\n",cudaGetErrorString(Error));
}

void Call_GPUFunction(){
	status= cusparseXcoo2csr(handle,cooRowIndex, nnz, m, csrRowPtr,CUSPARSE_INDEX_BASE_ZERO);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("coo2csr fail");
	}

	status= cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz,
				&done, descr, cooVal, csrRowPtr, cooColIndex, &y[0], &dzero, &x[0]);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("csrmv fail");
	}
}


void Send_To_Host(){
	cudaError_t Error;
	Error = cudaMemcpy(yHostPtr, y, (size_t)(n*sizeof(float)), cudaMemcpyDeviceToHost);
	printf("CUDA error(memcpy y->yHostPtr) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(xHostPtr, x, (size_t)(m*sizeof(float)), cudaMemcpyDeviceToHost);
        printf("CUDA error(memcpy x->xHostPtr) = %s\n",cudaGetErrorString(Error));
}

void Free_Memory(){
	if (yHostPtr) free(yHostPtr);
	if (xHostPtr) free(xHostPtr);
        if (cooRowIndexHostPtr) free(cooRowIndexHostPtr);
        if (cooColIndexHostPtr) free(cooColIndexHostPtr);
        if (cooValHostPtr) free(cooValHostPtr);
        if (y) cudaFree(y);
	if (x) cudaFree(x);
        if (csrRowPtr) cudaFree(csrRowPtr);
        if (cooRowIndex) cudaFree(cooRowIndex);
        if (cooColIndex) cudaFree(cooColIndex);
        if (cooVal) cudaFree(cooVal);
	if (cooValLU) cudaFree(cooValLU);
        if (descr) cusparseDestroyMatDescr(descr);
        if (handle) cusparseDestroy(handle);

}
