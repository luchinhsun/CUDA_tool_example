#include "head.h"

extern float * A;
extern float * B;
extern float * C;
extern float alpha, beta;

int main()
{
	int i, j;
        struct timeb start, end;
        int diff;

	Allocate_Memory();
	Init();
	printf("A = \n");
        for(i=0;i<m;i++){
                for(j=0;j<k;j++){
                        printf("%f ",A[i+j*m]);
                }
                printf("\n");
        }
	printf("B = \n");
        for(i=0;i<k;i++){
                for(j=0;j<n;j++){
                        printf("%f ",B[i+j*k]);
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

	//Save_Result();

	printf("C = \n");
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){
			printf("%f ",C[i+j*m]);
		}
		printf("\n");
	}

	printf("\n");
	Free_Memory();

}
