#include <stdio.h>
#include <malloc.h>
#include "head.h"

#define tpb 256

void Allocate(double **subT, double **subNUT, double **subMUT, double **subNDT, double **subMDT, double **d_subT, double **d_subBT, double **d_subNUT, double **d_subMUT, double **d_subNDT, double **d_subMDT){
	cudaError_t Error;
	*subT = (double *)malloc(subN*sizeof(double));
	*subNUT = (double *)malloc(n*sizeof(double));
	*subMUT = (double *)malloc(n*sizeof(double));
	*subNDT = (double *)malloc(n*sizeof(double));
	*subMDT = (double *)malloc(n*sizeof(double));
	
	cudaMalloc((void**)d_subT,subN*sizeof(double));
	cudaMalloc((void**)d_subBT,subBN*sizeof(double));
	cudaMalloc((void**)d_subNUT,n*sizeof(double));
	cudaMalloc((void**)d_subMUT,n*sizeof(double));
	cudaMalloc((void**)d_subNDT,n*sizeof(double));
	Error = cudaMalloc((void**)d_subMDT,n*sizeof(double));
	if (DEBUG)	printf("CUDA Error(malloc d_subMDT) = %s\n", cudaGetErrorString(Error));
}

void Send_To_Device(int phase, double **subT, double **subNUT, double **subNDT, double **d_subT, double **d_subNUT, double **d_subNDT){
	cudaError_t Error;
	if(phase == 0){
		Error = cudaMemcpy(*d_subT, *subT, subN*sizeof(double), cudaMemcpyHostToDevice);
		if (DEBUG)	printf("CUDA Error(copy subT->d_subT) = %s\n", cudaGetErrorString(Error));
	}
	if(phase == 1){
		Error = cudaMemcpy(*d_subNUT, *subNUT, n*sizeof(double), cudaMemcpyHostToDevice);
		if (DEBUG)	printf("CUDA Error(copy subNUT->d_subNUT) = %s\n", cudaGetErrorString(Error));
		Error = cudaMemcpy(*d_subNDT, *subNDT, n*sizeof(double), cudaMemcpyHostToDevice);
		if (DEBUG)	printf("CUDA Error(copy subNDT->d_subNDT) = %s\n", cudaGetErrorString(Error));
	}
}

void Send_To_Host(int phase, double **subT, double **subMUT, double **subMDT, double **d_subT, double **d_subMUT, double **d_subMDT){
	cudaError_t Error;
	if(phase == 0){
		Error = cudaMemcpy(*subT, *d_subT, subN*sizeof(double), cudaMemcpyDeviceToHost);
		if (DEBUG)	printf("CUDA Error(copy d_subT->subT) = %s\n", cudaGetErrorString(Error));
	}
	if(phase == 1){
		Error = cudaMemcpy(*subMUT, *d_subMUT, n*sizeof(double), cudaMemcpyDeviceToHost);
		if (DEBUG)	printf("CUDA Error(copy d_subMUT->subMUT) = %s\n", cudaGetErrorString(Error));
		Error = cudaMemcpy(*subMDT, *d_subMDT, n*sizeof(double), cudaMemcpyDeviceToHost);
		if (DEBUG)	printf("CUDA Error(copy d_subMDT->subMDT) = %s\n", cudaGetErrorString(Error));
	}
}

__global__ void boundary0(double *d_subT, double *d_subBT, double*d_subMUT, double *d_subMDT){
	int i = blockDim.x * blockIdx.x +threadIdx.x;
	int x, id;

	if(i<subN){
		x = i/n;
		id = i+(n+2)+1+2*x;
		d_subBT[id] = d_subT[i];
	}
	if(i<subn){
		d_subBT[(i+1)*(n+2)] = d_subT[i*n+n-1];
		d_subBT[(i+1)*(n+2)+n+1] = d_subT[i*n];
	}
	if(i<n){
		d_subMUT[i] = d_subT[i];
		d_subMDT[i] = d_subT[(subn-1)*n+i];
	}
}

__global__ void boundary1(double *d_subBT, double*d_subNUT, double *d_subNDT){
	int i = blockDim.x * blockIdx.x +threadIdx.x;
	
	if(i<n){
		d_subBT[i+1] = d_subNDT[i];
		d_subBT[(subn+1)*(n+2)+i+1] = d_subNUT[i];
	}
}

void CUDA_bdy(int phase, double **d_subT, double **d_subBT, double **d_subNUT, double **d_subMUT, double **d_subNDT, double **d_subMDT){
	int bpg0 = (subN+tpb-1)/tpb;
	int bpg1 = (n+tpb-1)/tpb;
	if(phase == 0)	boundary0<<<bpg0, tpb>>>(*d_subT, *d_subBT, *d_subMUT, *d_subMDT);
	if(phase == 1)	boundary1<<<bpg1, tpb>>>(*d_subBT, *d_subNUT, *d_subNDT);
	cudaDeviceSynchronize();
}

__global__ void Forward_Euler(double *d_subT, double *d_subBT){
	int i = blockDim.x * blockIdx.x +threadIdx.x;
	int id, x;
	
	if(i<subN){
		x = i/n;
		id = i+(n+2)+1+2*x;
		d_subT[i] = d_subBT[id] + 0.1*(d_subBT[id-(n+2)] + d_subBT[id+(n+2)] + d_subBT[id-1] + d_subBT[id+1]);
	}
			
}

void CUDA_FE(double **d_subT, double **d_subBT){
	int bpg = (subN+tpb-1)/tpb;
	Forward_Euler<<<bpg, tpb>>>(*d_subT, *d_subBT);
}

void Free(double **subT, double **subNUT, double **subMUT, double **subNDT, double **subMDT, double **d_subT, double **d_subBT, double **d_subNUT, double **d_subMUT, double **d_subNDT, double **d_subMDT){
	
	free(*subT);
	free(*subMUT);free(*subMDT);
	free(*subNUT);free(*subNDT);
	cudaFree(*d_subT);cudaFree(*d_subBT);
	cudaFree(*d_subNUT);cudaFree(*d_subMUT);
	cudaFree(*d_subNDT);cudaFree(*d_subMDT);
}
