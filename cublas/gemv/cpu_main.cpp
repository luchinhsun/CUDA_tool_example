#include "head.h"

extern float * A;
extern float * b;
extern float * c;
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
                for(j=0;j<n;j++){
                        printf("%f ",A[i+j*m]);
                }
                printf("\n");
        }
	printf("b = \n");
        for(i=0;i<n;i++){
                printf("%f\n",b[i]);
        }

	ftime(&start);
	Send_To_Device();
	Call_GPUFunction();
	Send_To_Host();
	ftime(&end);

        diff = (int)(1000.0*(end.time-start.time)+(end.millitm-start.millitm));
        printf("\nTime = %d ms\n", diff);

	//Save_Result();

	printf("c = \n");
	for(i=0;i<m;i++){
		printf("%f\n",c[i]);
	}

	printf("\n");
	Free_Memory();

}
