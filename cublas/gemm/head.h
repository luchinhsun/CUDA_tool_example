#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <sys/timeb.h>

#define m 3
#define k 4
#define n 2

void Allocate_Memory();
void Init();
void Free_Memory();
void Send_To_Device();
void Send_To_Host();
void Call_GPUFunction();
void Save_Result();
