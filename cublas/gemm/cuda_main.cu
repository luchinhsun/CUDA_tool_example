#include "head.h"

//variable for cublas
cublasStatus_t stat;
cublasHandle_t handle=0;

float * A;
float * d_A;
float * B;
float * d_B;
float * C;
float * d_C;
float alpha, beta;

void Allocate_Memory(){
	cudaError_t Error;

	A = (float *)malloc(m*k*sizeof(float));
	B = (float *)malloc(k*n*sizeof(float));
	C = (float *)malloc(m*n*sizeof(float));

	Error = cudaMalloc((void **)&d_A, m*k*sizeof(float));
	printf("CUDA error(malloc d_A) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void **)&d_B, k*n*sizeof(float));
        printf("CUDA error(malloc d_B) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void **)&d_C, m*n*sizeof(float));
        printf("CUDA error(malloc d_C) = %s\n",cudaGetErrorString(Error));
}

void Init(){
	int i, j;
	for(j=0;j<k;j++){
		for(i=0;i<m;i++){
			A[i+j*m] = int(rand()%10);
		}
	}
	for(j=0;j<n;j++){
		for(i=0;i<k;i++){
			B[i+j*k] = int(rand()%10);
		}
	}
	for(j=0;j<n;j++){
		for(i=0;i<m;i++){
			C[i+j*m] = 0.0;
		}
	}
	alpha = 1.0, beta = 0.0;
}

void Send_To_Device(){
	cudaError_t Error;
	Error = cudaMemcpy(d_A, A, m*k*sizeof(float), cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy A) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_B, B, k*n*sizeof(float), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy B) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_C, C, m*n*sizeof(float), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy C) = %s\n",cudaGetErrorString(Error));
}

void Call_GPUFunction(){
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("CUBLAS initialization failed\n");
	}

	cublasSgemm(  handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                d_A,
                m,
                d_B,
                k,
                &beta,
                d_C,
                m);
	//cudaDeviceSynchronize();
}


void Send_To_Host(){
	cudaError_t Error;
	Error = cudaMemcpy(C, d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost);
        printf("CUDA error(memcpy d_C->C) = %s\n",cudaGetErrorString(Error));
}

void Free_Memory(){
	if (A) free(A);
        if (d_A) cudaFree(d_A);
	if (B) free(B);
        if (d_B) cudaFree(d_B);
	if (C) free(C);
        if (d_C) cudaFree(d_C);

        if (handle) cublasDestroy(handle);

}

void Save_Result() {

        FILE *pFile;
        int i, j;

        // Save the matrix
	pFile = fopen("C.txt","w");
        // Save the matrix C
        for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
                	fprintf(pFile, "%g\t", C[i*n+j]);
		}
		fprintf(pFile, "\n");
        }
        fclose(pFile);
}

