#include "head.h"

//variable for cublas
cublasStatus_t stat;
cublasHandle_t handle=0;

float * A;
float * d_A;
float * b;
float * d_b;
float * c;
float * d_c;
float alpha, beta;

void Allocate_Memory(){
	cudaError_t Error;

	A = (float *)malloc(m*n*sizeof(float));
	b = (float *)malloc(n*sizeof(float));
	c = (float *)malloc(m*sizeof(float));

	Error = cudaMalloc((void **)&d_A, m*n*sizeof(float));
	printf("CUDA error(malloc d_A) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void **)&d_b, n*sizeof(float));
        printf("CUDA error(malloc d_b) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void **)&d_c, m*sizeof(float));
        printf("CUDA error(malloc d_c) = %s\n",cudaGetErrorString(Error));
}

void Init(){
	int i, j;
	for(j=0;j<n;j++){
		for(i=0;i<m;i++){
			A[i+j*m] = i+j*m;//int(rand()%10);
		}
	}
	for(i=0;i<n;i++){
		b[i] = int(rand()%10);
	}
	for(i=0;i<m;i++){
		c[i] = 0.0;
	}
	alpha = 1.0, beta = 0.0;
}

void Send_To_Device(){
	cudaError_t Error;
	Error = cudaMemcpy(d_A, A, m*n*sizeof(float), cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy A) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy b) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_c, c, m*sizeof(float), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy c) = %s\n",cudaGetErrorString(Error));
}

void Call_GPUFunction(){
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("CUBLAS initialization failed\n");
	}

	cublasSgemv(  handle,
		CUBLAS_OP_N,
                m,
                n,
                &alpha,
                d_A,
                m,
                d_b,
                1,
                &beta,
                d_c,
                1);
	//cudaDeviceSynchronize();
}


void Send_To_Host(){
	cudaError_t Error;
	Error = cudaMemcpy(c, d_c, m*sizeof(float), cudaMemcpyDeviceToHost);
        printf("CUDA error(memcpy d_c->c) = %s\n",cudaGetErrorString(Error));
}

void Free_Memory(){
	if (A) free(A);
        if (d_A) cudaFree(d_A);
	if (b) free(b);
        if (d_b) cudaFree(d_b);
	if (c) free(c);
        if (d_c) cudaFree(d_c);

        if (handle) cublasDestroy(handle);

}

void Save_Result() {

        FILE *pFile;
        int i;

        // Save the matrix
	pFile = fopen("c.txt","w");
        // Save the vector c
        for (i = 0; i < m; i++) {
		fprintf(pFile, "%g\t", c[i]);
		fprintf(pFile, "\n");
        }
        fclose(pFile);
}

