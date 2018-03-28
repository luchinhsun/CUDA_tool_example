#include <stdio.h>
#include <malloc.h>
#include <sys/timeb.h>

#define n 1000
#define N (n*n)
#define BN (n+2)*(n+2)

double *T;
double *BT;

void Save_Result();

int main(){

	//time recorder
	struct timeb start, end;
	int diff;
	ftime(&start);

	int i, k;
	int x, id;
	//Allocate memory
	T = (double *)malloc(N*sizeof(double));
	BT = (double *)malloc(BN*sizeof(double));
	
	//Initial
	for(i=0;i<N;i++){
		if(i<n){
			T[i] = 1.0;
		}else{
			T[i] = 0.0;
		}
	}
	
	int ncount;
	ncount = 500;

	//FDM
	for(k=0;k<ncount;k++){
		for(i=0;i<N;i++){
			x = i/n;
			id = i+(n+2)+1+2*x;
			BT[id] = T[i];
		}
		for(i=0;i<n;i++){
			BT[i+1] = T[(n-1)*n+i];
			BT[(n+1)*(n+2)+i+1] = T[i];
			BT[(i+1)*(n+2)] = T[i*n+n-1];
			BT[(i+1)*(n+2)+n+1] = T[i*n];
		}
		for(i=0;i<N;i++){
			x = i/n;
			id = i+(n+2)+1+2*x;
			T[i] = BT[id] + 0.1*(BT[id-(n+2)] + BT[id+(n+2)] + BT[id-1] + BT[id+1]);
		}
	}
	
	ftime(&end);
	diff = (1000.0*(end.time - start.time) + (end.millitm - start.millitm));
	printf("%d ms\n", diff);
	Save_Result();
	
	free(T);free(BT);
	return 0;
}
void Save_Result(){

        FILE *pFile;
        int i,j;
        int index;
        //int n;
        //n = nx;
        pFile = fopen("sT.txt","w+");
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
