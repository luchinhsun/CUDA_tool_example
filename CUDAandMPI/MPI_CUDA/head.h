#define n 1000
#define N (n*n)
#define DEBUG 0		// Print debug messages (set to 0 to turn off)
#define subn ((int)(n-1)/2+1)
#define subN (subn)*n
#define subBN (subn+2)*(n+2)

void Save_Result(double *T);
void Allocate(double **subT, double **subNUT, double **subMUT, double **subNDT, double **subMDT, double **d_subT, double **d_subBT, double **d_subNUT, double **d_subMUT, double **d_subNDT, double **d_subMDT);
void Send_To_Device(int phase, double **subT, double **subNUT, double **subNDT, double **d_subT, double **d_subNUT, double **d_subNDT);
void CUDA_bdy(int phase, double **d_subT, double **d_subBT, double **d_subNUT, double **d_subMUT, double **d_subNDT, double **d_subMDT);
void CUDA_FE(double **d_subT, double **d_subBT);
void Send_To_Host(int phase, double **subT, double **subMUT, double **subMDT, double **d_subT, double **d_subMUT, double **d_subMDT);
void Free(double **subT, double **subNUT, double **subMUT, double **subNDT, double **subMDT, double **d_subT, double **d_subBT, double **d_subNUT, double **d_subMUT, double **d_subNDT, double **d_subMDT);