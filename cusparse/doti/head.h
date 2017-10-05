#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cusparse.h>
#include <sys/timeb.h>

#define n 4
#define nnz 2

void Allocate_Memory_and_Init();
void Free_Memory();
void Send_To_Device();
void Send_To_Host();
void Call_GPUFunction();
