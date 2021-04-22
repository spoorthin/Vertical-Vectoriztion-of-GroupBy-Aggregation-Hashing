#include <immintrin.h>
#include <stdlib.h>
#include <iostream>
#include <emmintrin.h>
#include <smmintrin.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>

int VectorProbeInsert(int arr[], int datasize,int c);
unsigned int hash1(int key);
unsigned int hash2(int key);
void print128_num(__m512i var);
void insert(int arr[], int hv[], int cnt[]);
void clearHash();
int hashCheck();
int addValues();
long double  getSIMDTime();
void clearClocks();
