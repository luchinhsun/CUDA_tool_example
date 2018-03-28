#include <stdio.h>
#include <malloc.h>
#include <sys/timeb.h>
#include <mpi.h>

#define n 1000
#define N (n*n)
//#define BN (n+2)*(n+2)

#define subn ((int)(n-1)/2+1)
#define subN (subn)*n
#define subBN (subn+2)*(n+2)
/*
double *T;
//double *BT;

double *subT;
double *subBT;
double *subNUT;
double *subMUT;
double *subNDT;
double *subMDT;
*/
void Save_Result(double *T);
void function(int phase, double **T, double *subT);

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
	double *T;
	//time recorder
    struct timeb start, end;
    int diff;
    ftime(&start);
	int i, k;
	int x, id;
	//Allocate memory & Initial
	if (world_rank == 0){
		//time recorder
		//struct timeb start, end;
		//int diff;
		//ftime(&start);

		//double *T;
		T = (double *)malloc(N*sizeof(double));
		//BT = (double *)malloc(BN*sizeof(double));

		for(i=0;i<N;i++){
			//x = i/n;
			//id = i+(n+2)+1+2*x;
			if(i<n){
				T[i] = 1.0;
			}else{
				T[i] = 0.0;
			}
			//BT[id] = T[i];
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	double *subT;
	double *subBT;
	double *subNUT;
	double *subMUT;
	double *subNDT;
	double *subMDT;
	subT = (double *)malloc(subN*sizeof(double));
	subBT = (double *)malloc(subBN*sizeof(double));
	subNUT = (double *)malloc(n*sizeof(double));
	subMUT = (double *)malloc(n*sizeof(double));
	subNDT = (double *)malloc(n*sizeof(double));
	subMDT = (double *)malloc(n*sizeof(double));

	int ncount;
	ncount = 500;

	MPI_Scatter(T, subN, MPI_DOUBLE, subT, subN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//MPI_Scatter(BT, subBN, MPI_DOUBLE, subBT, subBN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//FDM
	for(k=0;k<ncount;k++){
		for(i=0;i<subN;i++){
			x = i/n;
			id = i+(n+2)+1+2*x;
			subBT[id] = subT[i];
		}
		for(i=0;i<subn;i++){
			subBT[(i+1)*(n+2)] = subT[i*n+n-1];
			subBT[(i+1)*(n+2)+n+1] = subT[i*n];
		}
		for(i=0;i<n;i++){
			subMUT[i] = subT[i];
			subMDT[i] = subT[(subn-1)*n+i];;
		}
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
		for(i=0;i<n;i++){
			subBT[i+1] = subNDT[i];
			subBT[(subn+1)*(n+2)+i+1] = subNUT[i];
		}
		for(i=0;i<subN;i++){
			x = i/n;
			id = i+(n+2)+1+2*x;
			subT[i] = subBT[id] + 0.1*(subBT[id-(n+2)] + subBT[id+(n+2)] + subBT[id-1] + subBT[id+1]);
		}
	}

	MPI_Gather(subT, subN, MPI_DOUBLE, T, subN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//	if(world_rank == 1)	Save_Result();

	ftime(&end);
        diff = (1000.0*(end.time - start.time) + (end.millitm - start.millitm));
	if(world_rank == 0){
		//ftime(&end);
		//diff = (1000.0*(end.time - start.time) + (end.millitm - start.millitm));
		printf("%d ms\n", diff);
		Save_Result(T);
		free(T);//free(BT);
	}
	free(subT);free(subBT);
	free(subMUT);free(subMDT);
	free(subNUT);free(subNDT);
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
