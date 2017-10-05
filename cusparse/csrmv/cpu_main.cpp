#include "head.h"

extern float * yHostPtr;
extern float * xHostPtr;
extern int *cooRowIndexHostPtr;
extern int * cooColIndexHostPtr;
extern float * cooValHostPtr;

int main()
{
	int i, j;
	struct timeb start, end;
        int diff;

	Allocate_Memory_and_Init();

	float A[m*n];
	for(i=0;i<m*n;i++){
		A[i] = 0.0;
	}
	for(i=0;i<nnz;i++){
		A[cooRowIndexHostPtr[i]*n+cooColIndexHostPtr[i]] = cooValHostPtr[i];
	}
	printf("A = \n");
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){
			printf("%g, ", A[i*n+j]);
		}
		printf("\n");
	}


	ftime(&start);
	Send_To_Device();

	Call_GPUFunction();

	Send_To_Host();
	ftime(&end);
        diff = (int)(1000.0*(end.time-start.time)+(end.millitm-start.millitm));
        printf("\nTime = %d ms\n", diff);

	printf("y = ");
	for(i=0;i<n;i++){
		printf("%g, ",yHostPtr[i]);
	}
	printf("\n");

	printf("A*y = ");
        for(i=0;i<m;i++){
                printf("%g, ",xHostPtr[i]);
        }

	printf("\n");
	Free_Memory();

	return 0;
}
