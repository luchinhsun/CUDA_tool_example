#include "head.h"

//variable for cublas
cublasStatus_t stat;
cublasHandle_t handle=0;

float * A;
float * d_A;
float * b;
float * d_b;
float * C;
float * d_C;
float * x;
float * d_x;
float alpha, beta;

#define batchsize 1
int * PivotA;
int * infoA;
int * h_PivotA;
int * h_infoA;

float ** h_Apoint;
float ** d_Apoint;
float ** h_Cpoint;
float ** d_Cpoint;


void Allocate_Memory(){
	cudaError_t Error;

	A = (float *)malloc(m*n*sizeof(float));
	b = (float *)malloc(n*sizeof(float));
	x = (float *)malloc(n*sizeof(float));
	C = (float *)malloc(m*n*sizeof(float));

	Error = cudaMalloc((void **)&d_A, m*n*sizeof(float));
	printf("CUDA error(malloc d_A) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void **)&d_b, n*sizeof(float));
        printf("CUDA error(malloc d_b) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void **)&d_x, n*sizeof(float));
        printf("CUDA error(malloc d_x) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void **)&d_C, m*n*sizeof(float));
        printf("CUDA error(malloc d_C) = %s\n",cudaGetErrorString(Error));

	h_PivotA = (int *)malloc(n*batchsize*sizeof(int));
	h_infoA = (int *)malloc(batchsize*sizeof(int));
	Error = cudaMalloc((void **)&PivotA, n*batchsize*sizeof(int));
        printf("CUDA error(malloc PivotA) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void **)&infoA, batchsize*sizeof(int));
        printf("CUDA error(malloc infoA) = %s\n",cudaGetErrorString(Error));

	h_Apoint = (float **)malloc(batchsize*sizeof(float*));
	Error = cudaMalloc((void **)&d_Apoint, batchsize*sizeof(float*));
        printf("CUDA error(malloc d_Apoint) = %s\n",cudaGetErrorString(Error));
	h_Cpoint = (float **)malloc(batchsize*sizeof(float*));
        Error = cudaMalloc((void **)&d_Cpoint, batchsize*sizeof(float*));
        printf("CUDA error(malloc d_Cpoint) = %s\n",cudaGetErrorString(Error));
}

void Init(){
	int i, j;
	for(j=0;j<n;j++){
		for(i=0;i<m;i++){
			A[i+j*m] = int(rand()%10);
		}
	}
	for(i=0;i<n;i++){
		b[i] = int(rand()%10);
		x[i] = 0.0;
	}
	for(i=0;i<m*n;i++){
		C[i] = 0.0;
	}
	alpha = 1.0, beta = 0.0;

	for(i=0;i<batchsize;i++){
		h_Apoint[0] = d_A + i*m*n;
		h_Cpoint[0] = (float *)((char*)d_C+i*((size_t)m*n)*sizeof(float));
	}
}

void Send_To_Device(){
	cudaError_t Error;
	Error = cudaMemcpy(d_A, A, m*n*sizeof(float), cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy A) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy b) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_C, C, m*n*sizeof(float), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy C) = %s\n",cudaGetErrorString(Error));

	Error = cudaMemcpy(d_Apoint, h_Apoint, batchsize*sizeof(float*), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy Apoint) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_Cpoint, h_Cpoint, batchsize*sizeof(float*), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy Cpoint) = %s\n",cudaGetErrorString(Error));
}

void Call_GPUFunction(){
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("CUBLAS initialization failed\n");
	}

	cublasSgetrfBatched( handle,
		m,
		d_Apoint,
		m,
		PivotA,
		infoA,
		batchsize);

	cublasSgetriBatched( handle,
		m,
		(const float **)d_Apoint,
		m,
		PivotA,
		d_Cpoint,
		m,
		infoA,
		batchsize);

	cublasSgemv(  handle,
                CUBLAS_OP_N,
                m,
                n,
                &alpha,
                d_C,
                m,
                d_b,
                1,
                &beta,
                d_x,
                1);

	cudaDeviceSynchronize();
}


void Send_To_Host(){
	cudaError_t Error;
	Error = cudaMemcpy(x, d_x, n*sizeof(float), cudaMemcpyDeviceToHost);
        printf("CUDA error(memcpy d_x->x) = %s\n",cudaGetErrorString(Error));

	Error = cudaMemcpy(C, d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost);
        printf("CUDA error(memcpy d_C->C) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(A, d_A, m*n*sizeof(float), cudaMemcpyDeviceToHost);
        printf("CUDA error(memcpy d_A->A) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(h_PivotA, PivotA, n*batchsize*sizeof(int), cudaMemcpyDeviceToHost);
        printf("CUDA error(memcpy PivotA) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(h_infoA, infoA, batchsize*sizeof(int), cudaMemcpyDeviceToHost);
        printf("CUDA error(memcpy infoA) = %s\n",cudaGetErrorString(Error));
}

void Free_Memory(){
	if (A) free(A);
        if (d_A) cudaFree(d_A);
	if (b) free(b);
        if (d_b) cudaFree(d_b);
	if (x) free(x);
        if (d_x) cudaFree(d_x);
	if (C) free(C);
        if (d_C) cudaFree(d_C);

	if (h_PivotA) free(h_PivotA);
	if (PivotA) cudaFree(PivotA);
	if (h_infoA) free(h_infoA);
	if (infoA) cudaFree(infoA);

	if (h_Apoint) free(h_Apoint);
	if (d_Apoint) cudaFree(d_Apoint);
	if (h_Cpoint) free(h_Cpoint);
        if (d_Cpoint) cudaFree(d_Cpoint);

        if (handle) cublasDestroy(handle);

}

void Save_Result() {

        FILE *pFile;
        int i;

        // Save the matrix
	pFile = fopen("x.txt","w");
        // Save the vector x
        for (i = 0; i < n; i++) {
		fprintf(pFile, "%g\t", x[i]);
		fprintf(pFile, "\n");
        }
        fclose(pFile);
}

