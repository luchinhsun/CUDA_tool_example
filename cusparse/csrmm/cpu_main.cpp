#include "head.h"

extern float * yHostPtr;
extern float * xHostPtr;

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

	Save_Result();

	printf("\n");
	Free_Memory();

}
