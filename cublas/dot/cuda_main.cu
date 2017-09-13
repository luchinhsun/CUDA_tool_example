#include "head.h"

//variable for cublas
cublasStatus_t stat;
cublasHandle_t handle=0;

float * a;
float * d_a;
float * b;
float * d_b;
float * c;
float * d_c;

void Allocate_Memory(){
	cudaError_t Error;

	a = (float *)malloc(n*sizeof(float));
	b = (float *)malloc(n*sizeof(float));
	c = (float *)malloc(1*sizeof(float));

	Error = cudaMalloc((void **)&d_a, n*sizeof(float));
	printf("CUDA error(malloc d_a) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void **)&d_b, n*sizeof(float));
        printf("CUDA error(malloc d_b) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void **)&d_c, 1*sizeof(float));
        printf("CUDA error(malloc d_c) = %s\n",cudaGetErrorString(Error));
}

void Init(){
	int i;
	for(i=0;i<n;i++){
		a[i] = i;//int(rand()%10);
	}
	for(i=0;i<n;i++){
		b[i] = int(rand()%10);
	}
	c[0] = 0.0;
}

void Send_To_Device(){
	cudaError_t Error;
	Error = cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy A) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy b) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_c, c, 1*sizeof(float), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy c) = %s\n",cudaGetErrorString(Error));
}

void Call_GPUFunction(){
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("CUBLAS initialization failed\n");
	}

	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

	cublasSdot(  handle,
                n,
                d_a,
                1,
                d_b,
                1,
                d_c);
	//cudaDeviceSynchronize();

}


void Send_To_Host(){
	cudaError_t Error;
	Error = cudaMemcpy(c, d_c, 1*sizeof(float), cudaMemcpyDeviceToHost);
        printf("CUDA error(memcpy d_c->c) = %s\n",cudaGetErrorString(Error));
}

void Free_Memory(){
	if (a) free(a);
        if (d_a) cudaFree(d_a);
	if (b) free(b);
        if (d_b) cudaFree(d_b);
	if (c) free(c);
        if (d_c) cudaFree(d_c);

        if (handle) cublasDestroy(handle);

}

void Save_Result() {

        FILE *pFile;
        int i;

        // Save the vector
	pFile = fopen("c.txt","w");
        // Save the vector c
        for (i = 0; i < 1; i++) {
		fprintf(pFile, "%g\t", c[i]);
        }
        fclose(pFile);
}

