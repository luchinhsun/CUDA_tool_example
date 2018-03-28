#include <stdio.h>
#include <malloc.h>
#include <sys/timeb.h>
#include <mpi.h>
#include "head.h"

int main(){
	// Initialize the MPI environment
    MPI_Init(NULL, NULL);
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

	//time recorder
	struct timeb start, end;
	int diff;
	ftime(&start);
	
	double *T;

	double *subT;
	double *subNUT;
	double *subMUT;
	double *subNDT;
	double *subMDT;

	double *d_subT;
	double *d_subBT;
	double *d_subNUT;
	double *d_subMUT;
	double *d_subNDT;
	double *d_subMDT;
	
	int i, k;
	//Allocate memory & Initial
	if (world_rank == 0){
		T = (double *)malloc(N*sizeof(double));
		for(i=0;i<N;i++){
			if(i<n){
				T[i] = 1.0;
			}else{
				T[i] = 0.0;
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	Allocate(&subT, &subNUT, &subMUT, &subNDT, &subMDT,	&d_subT, &d_subBT, &d_subNUT, &d_subMUT, &d_subNDT, &d_subMDT);

	int ncount;
	ncount = 500;

	MPI_Scatter(T, subN, MPI_DOUBLE, subT, subN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	Send_To_Device(0, &subT, &subNUT, &subNDT, &d_subT, &d_subNUT, &d_subNDT);
	
	//FDM
	for(k=0;k<ncount;k++){
		CUDA_bdy(0, &d_subT, &d_subBT, &d_subNUT, &d_subMUT, &d_subNDT, &d_subMDT);
		Send_To_Host(1, &subT, &subMUT, &subMDT, &d_subT, &d_subMUT, &d_subMDT);
		if(world_rank == 0){
			MPI_Send(subMUT, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
			MPI_Recv(subNDT, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(subMDT, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
			MPI_Recv(subNUT, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}else if(world_rank == 1){
			MPI_Recv(subNUT, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(subMDT, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			MPI_Recv(subNDT, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(subMUT, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
		Send_To_Device(1, &subT, &subNUT, &subNDT, &d_subT, &d_subNUT, &d_subNDT);
		CUDA_bdy(1, &d_subT, &d_subBT, &d_subNUT, &d_subMUT, &d_subNDT, &d_subMDT);
		CUDA_FE(&d_subT, &d_subBT);
	}
	
	Send_To_Host(0, &subT, &subMUT, &subMDT, &d_subT, &d_subMUT, &d_subMDT);
	MPI_Gather(subT, subN, MPI_DOUBLE, T, subN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	ftime(&end);
	diff = (1000.0*(end.time - start.time) + (end.millitm - start.millitm));

	if(world_rank == 0){
		printf("%d ms\n", diff);
		Save_Result(T);
		free(T);//free(BT);
	}
	Free(&subT, &subNUT, &subMUT, &subNDT, &subMDT,	&d_subT, &d_subBT, &d_subNUT, &d_subMUT, &d_subNDT, &d_subMDT);
	MPI_Finalize();
	return 0;
}
void Save_Result(double *T){

        FILE *pFile;
        int i,j;
        int index;
        //int n;
        //n = nx;
        pFile = fopen("T.txt","w+");
        // Save the matrix V
        for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                        index = i*n + j;
                        fprintf(pFile, "%g", T[index]);
                        if (j == (n-1)) {
                                fprintf(pFile, "\n");
                        }else{
                                fprintf(pFile, "\t");
                        }
                }
        }
        fclose(pFile);
}
