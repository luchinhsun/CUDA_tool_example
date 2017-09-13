#include "head.h"

extern float * a;
extern float * b;
extern float * c;

int main()
{
	int i, j;
        struct timeb start, end;
        int diff;

	Allocate_Memory();
	Init();
	printf("a = \n");
        for(i=0;i<n;i++){
                printf("%f ",a[i]);
        }
	printf("\nb = \n");
        for(i=0;i<n;i++){
                printf("%f ",b[i]);
        }
	printf("\n");

	ftime(&start);
	Send_To_Device();
	Call_GPUFunction();
	Send_To_Host();
	ftime(&end);

        diff = (int)(1000.0*(end.time-start.time)+(end.millitm-start.millitm));
        printf("\nTime = %d ms\n", diff);

	//Save_Result();

	printf("\nc = %f\n",c[0]);

	printf("\n");
	Free_Memory();

}
