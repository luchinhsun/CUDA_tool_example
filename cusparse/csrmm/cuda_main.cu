#include "head.h"

//variable for cusparse
cusparseStatus_t status;
cusparseHandle_t handle=0;
cusparseMatDescr_t descr=0;
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
float * A;

float dzero =0.0;
float done =1.0;
float dtwo =2.0;
float dthree=3.0;
float dfive =5.0;

void Allocate_Memory_and_Init(){
	//cusparse
	size_t size = nnz*sizeof(int);
	cooRowIndexHostPtr = (int *) malloc(size);
	cooColIndexHostPtr = (int *) malloc(size);
	cooValHostPtr = (float *)malloc(nnz*sizeof(float));

	cooRowIndexHostPtr[0] = 0;cooColIndexHostPtr[0]=1;cooValHostPtr[0]=1.0;
	cooRowIndexHostPtr[1] = 0;cooColIndexHostPtr[1]=2;cooValHostPtr[1]=2.0;
	cooRowIndexHostPtr[2] = 1;cooColIndexHostPtr[2]=0;cooValHostPtr[2]=3.0;
	cooRowIndexHostPtr[3] = 1;cooColIndexHostPtr[3]=1;cooValHostPtr[3]=4.0;
	cooRowIndexHostPtr[4] = 1;cooColIndexHostPtr[4]=2;cooValHostPtr[4]=5.0;
	cooRowIndexHostPtr[5] = 2;cooColIndexHostPtr[5]=1;cooValHostPtr[5]=6.0;
	cooRowIndexHostPtr[6] = 2;cooColIndexHostPtr[6]=3;cooValHostPtr[6]=7.0;
	cooRowIndexHostPtr[7] = 3;cooColIndexHostPtr[7]=1;cooValHostPtr[7]=8.0;
	cooRowIndexHostPtr[8] = 4;cooColIndexHostPtr[8]=2;cooValHostPtr[8]=9.0;

	int i;
	A	= (float *)malloc(m*k*sizeof(float));
	yHostPtr    = (float *)malloc(k*n*sizeof(float));
	for (i=0;i<k*n;i++){
		yHostPtr[i] = i;
	}

	xHostPtr    = (float *)malloc(m*n*sizeof(float));

	cudaError_t Error;

	Error = cudaMalloc((void**)&cooRowIndex, size);
	printf("CUDA error(malloc RowIndex) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&cooColIndex, size);
	printf("CUDA error(malloc ColIndex) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&cooVal, nnz*sizeof(float));
	printf("CUDA error(malloc Val) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&cooValLU, nnz*sizeof(float));
        printf("CUDA error(malloc Val) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&y, k*n*sizeof(float));
	printf("CUDA error(malloc y) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&x, m*n*sizeof(float));
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
	Error = cudaMemcpy(y, yHostPtr, (size_t)(k*n*sizeof(float)), cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy y) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(x, xHostPtr, (size_t)(m*n*sizeof(float)), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy x) = %s\n",cudaGetErrorString(Error));
}

void Call_GPUFunction(){
	status= cusparseXcoo2csr(handle,cooRowIndex,nnz,m, csrRowPtr,CUSPARSE_INDEX_BASE_ZERO);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("coo2csr fail");
	}

	status= cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, nnz, &done, descr,
                                                cooVal, csrRowPtr, cooColIndex, y, k, &done, x, m);
}


void Send_To_Host(){
	cudaError_t Error;
	Error = cudaMemcpy(yHostPtr, y, (size_t)(k*n*sizeof(float)), cudaMemcpyDeviceToHost);
	printf("CUDA error(memcpy y->yHostPtr) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(xHostPtr, x, (size_t)(m*n*sizeof(float)), cudaMemcpyDeviceToHost);
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
	if (temp) cudaFree(temp);
        if (csrRowPtr) cudaFree(csrRowPtr);
        if (cooRowIndex) cudaFree(cooRowIndex);
        if (cooColIndex) cudaFree(cooColIndex);
        if (cooVal) cudaFree(cooVal);
	if (cooValLU) cudaFree(cooValLU);
        if (descr) cusparseDestroyMatDescr(descr);
        if (handle) cusparseDestroy(handle);
        if (A) free(A);
}

void Save_Result() {

        FILE *pFile;
        int i, j;

        // Save the matrix A
	for(i=0;i<m*k;i++){
                A[i] = 0.0;
        }
        for(i=0;i<nnz;i++){
                A[cooRowIndexHostPtr[i]*k+cooColIndexHostPtr[i]] = cooValHostPtr[i];
        }
        pFile = fopen("A.txt","w");
        for (i = 0; i < m; i++) {
                for (j = 0; j < k; j++) {
                        fprintf(pFile, "%g\t", A[i*k+j]);
                }
                fprintf(pFile, "\n");
        }
        fclose(pFile);

	pFile = fopen("B.txt","w");
        // Save the matrix B
        for (i = 0; i < k; i++){
		for (j = 0; j<n; j++){
                	fprintf(pFile, "%g\t", yHostPtr[i+j*k]);
		}
		fprintf(pFile, "\n");
        }
        fclose(pFile);

        pFile = fopen("X.txt","w");
        // Save the matrix x
        for (i = 0; i < m; i++) {
		for (j = 0; j<n; ++j){
                	fprintf(pFile, "%g\t", xHostPtr[i+j*m]);
		}
                fprintf(pFile, "\n");
        }
        fclose(pFile);
}
