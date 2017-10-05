#include "head.h"

extern float * yHostPtr;
extern float * xHostPtr;
extern float * Hostresult;
extern int * xHostInd;

int main()
{
	int i;
	struct timeb start, end;
        int diff;

	Allocate_Memory_and_Init();

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

	printf("x = ");
        for(i=0;i<nnz;i++){
                printf("%g, ",xHostPtr[i]);
        }

	printf("xInd = ");
        for(i=0;i<nnz;i++){
                printf("%d, ",xHostInd[i]);
        }

	printf("Hostresult = %g", Hostresult[0]);

	printf("\n");
	Free_Memory();

	return 0;
}
