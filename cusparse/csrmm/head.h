#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cusparse_v2.h>
#include <sys/timeb.h>

#define m 5
#define n 3
#define k 4
#define nnz 9

void Allocate_Memory_and_Init();
void Free_Memory();
void Send_To_Device();
void Send_To_Host();
void Call_GPUFunction();
void Save_Result();
