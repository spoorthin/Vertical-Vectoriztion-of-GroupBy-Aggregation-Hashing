#include <immintrin.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <stdbool.h>
#include "LinearProbingVertical.h"

#define KEYSPERPAGE 1
#define REPEATS 2
#define HSIZE 50000000

clock_t SIMDInsertBegin, SIMDInsertEnd;
static long double SIMDTime;

int hashtablePay[HSIZE];
int hashtableKeys[HSIZE];

unsigned int hash(int key){
    return ((unsigned long)((unsigned int)1300000077*key)* HSIZE)>>32;
}


void print128_num(__m512i var){
    int *resi = (int*) &var;
    std::cout << resi[0] << resi[1] << resi[2] << resi[3] << resi[4] << resi[5] << resi[6]
    << resi[7] << resi[8] << resi[9] << resi[10] << resi[11] << resi[12] << resi[13] << resi[14] << resi[15];
}


void insert(int arr[], int hv[], int cnt[]) {
    //This function inserts keys that were invalidated from input lanes
    __mmask16 m,j;
    int i=0,mask;

    //set input vector, hash vector and number of occurrence of each key in the input vector
    __m512i inputVector = _mm512_setr_epi32 (arr[i], arr[i+1], arr[i+2], arr[i+3], arr[i+4], arr[i+5], arr[i+6],arr[i+7], arr[i+8], arr[i+9], arr[i+10], arr[i+11], arr[i+12], arr[i+13],arr[i+14],arr[i+15]);
    __m512i hashVector = _mm512_setr_epi32(hv[i], hv[i+1], hv[i+2], hv[i+3], hv[i+4], hv[i+5],hv[i+6], hv[i+7],hv[i+8], hv[i+9], hv[i+10], hv[i+11], hv[i+12], hv[i+13],hv[i+14],hv[i+15]);
    __m512i count = _mm512_setr_epi32(cnt[i],cnt[i+1], cnt[i+2], cnt[i+3], cnt[i+4], cnt[i+5], cnt[i+6], cnt[i+7],cnt[i+8], cnt[i+9], cnt[i+10], cnt[i+11], cnt[i+12], cnt[i+13], cnt[i+14], cnt[i+15]);
    __m512i zero = _mm512_setzero_epi32();

    do {
            __mmask16 a = _mm512_cmpneq_epi32_mask(hashVector,zero);
            __m512i setone = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);

            __m512i con = _mm512_mask_conflict_epi32(con,a,hashVector);
            __mmask16 a1 = _mm512_cmpeq_epi32_mask(con,zero);
            __mmask16 n =  _mm512_knot (a1);
            hashVector = _mm512_mask_add_epi32 (hashVector,n,hashVector, setone);


            //reset offsets if the end of hashtable is reached
            int H = HSIZE-1;
            __m512i setend = _mm512_setr_epi32 (H,H,H,H,H,H,H,H,H,H,H,H,H,H,H,H);
            __mmask16 end = _mm512_cmpeq_epi32_mask(setend, hashVector);
            hashVector = _mm512_mask_set1_epi32(hashVector,end,0);

            //increment offset
            hashVector = _mm512_mask_add_epi32 (hashVector,a,hashVector, setone);


            //gather keys and compare, if there is a match increment payload and scatter
            __m512i key_H = _mm512_i32gather_epi32(hashVector, &(hashtableKeys), 4);
            __mmask16 m1 = _mm512_cmpeq_epi32_mask(inputVector, key_H);
            __m512i payload = _mm512_i32gather_epi32(hashVector, &(hashtablePay), 4);
            __m512i incr = _mm512_mask_add_epi32(incr,m1,payload,count);
            _mm512_mask_i32scatter_epi32 (&(hashtablePay),m1,hashVector, incr, 4);

            inputVector = _mm512_mask_set1_epi32 (inputVector, m1, 0);

            //insert keys if there is empty slot in the hashtable
            __m512i key_HH = _mm512_i32gather_epi32(hashVector, &(hashtableKeys), 4);
            __mmask16 m2 = _mm512_cmpeq_epi32_mask(zero, key_HH);
            _mm512_mask_i32scatter_epi32 (&(hashtableKeys), m2,hashVector, inputVector, 4);
            _mm512_mask_i32scatter_epi32 (&(hashtablePay),m2,hashVector, count, 4);

            //set finished keys to zero
            inputVector = _mm512_mask_set1_epi32 (inputVector, m2, 0);
            j =  _mm512_cmpeq_epi32_mask(zero, inputVector);

            mask = (int) j; //loop until all input vector keys are zero
            if(mask==65535)
              break;
    }  while(1);

}

void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

int partition (int arr[], int low, int high)
{
    int pivot = arr[high];    // pivot
    int i = (low - 1);  // Index of smaller element
   #pragma omp parallel for
    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr, low, high);
        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}


int VectorProbeInsert(int arr[], int datasize,int c) {

    __mmask16  m;
    __m512i inputVector,hashVector;
    __m512i zero = _mm512_setzero_epi32();

    int *count = (int *)calloc(datasize,sizeof(int));
    int *input = (int *)calloc(datasize,sizeof(int));
    int mask,countN=0;
    
    if(c==2 || c==3) {
        quickSort(arr, 0, datasize-1);
        int prev = arr[0];
        int cnt = 1;
        
        #pragma omp parallel for
        for (int i = 1; i < datasize; i++) {
            if (arr[i] == prev) {
            cnt++;
        } else {
            input[i] = prev;
            count[i] = cnt;
            prev = arr[i];
            cnt = 1;
            }
        }
        input[datasize-1] = prev;
        count[datasize-1] = cnt;
    }
    if(c==0 || c==1) {
        for(int k=0;k<datasize;k++) {
                input[k] = arr[k];
                count[k]=1;
        }
    }

    #pragma omp parallel for
        for(int i=0;i<datasize;i+=16) {

        //set 16 input keys each time into input vector
        inputVector = _mm512_setr_epi32 (input[i], input[i+1], input[i+2], input[i+3], input[i+4], input[i+5], input[i+6],input[i+7], input[i+8], input[i+9], input[i+10], input[i+11], input[i+12], input[i+13],input[i+14],input[i+15]);

        int *hv = (int*) &hashVector;
        int *iv = (int*) &inputVector;

        __m512i countVector =  _mm512_setr_epi32 (count[i], count[i+1],count[i+2], count[i+3],count[i+4], count[i+5], count[i+6],count[i+7], count[i+8], count[i+9],count[i+10], count[i+11], count[i+12], count[i+13],count[i+14],count[i+15]);

        //set hash codes for 16 input keys
        #pragma omp parallel for
        for(int k=0;k<16;k++) {
            hv[k] = hash(iv[k]);
        }

        //gather the keys present in hashtable
        __m512i  key_H = _mm512_i32gather_epi32(hashVector, &(hashtableKeys), 4);


        //compare input vector keys with gathered keys of hashtable
        m = _mm512_cmpeq_epi32_mask(inputVector, key_H);
        mask = _mm512_mask2int(m);

        //if mask is 65535 that is all equal bits are set, i.e all input keys are a match
        if(mask==65535) {
            __m512i payload = _mm512_i32gather_epi32 (hashVector, &(hashtablePay), 4);
            __m512i incri = _mm512_add_epi32 (payload, countVector);
            _mm512_i32scatter_epi32 (&(hashtablePay), hashVector, incri, 4);
            continue;
        }

        m = _mm512_cmpeq_epi32_mask(key_H, zero);
        mask = _mm512_mask2int(m);
        if(mask == 65535 ){
            SIMDInsertBegin = clock();
            __m512i con1 = _mm512_conflict_epi32(hashVector);
            __mmask16 a1 = _mm512_cmpeq_epi32_mask(con1,zero);
            if((int)a1!=65535){
                //first scatter non-conflicting keys
                _mm512_mask_i32scatter_epi32 (&(hashtableKeys),a1,hashVector,inputVector, 4);
                _mm512_mask_i32scatter_epi32 (&(hashtablePay),a1, hashVector,countVector, 4);
                //gather again to check how many are inserted
                __m512i gthr = _mm512_i32gather_epi32(hashVector, &(hashtableKeys), 4);
                __mmask16 a2 = _mm512_cmpeq_epi32_mask(gthr,inputVector);
                //if still there are collisions add them to stack and perfor linear probing
                if((int)a2!=65535) {
                    int stackInput[16]= {0};
                    int stackHash[16] = {0};
                    int conPay[16]={0};
                    int *iv = (int*) &inputVector;
                    int *ct = (int*) &countVector;
                    int *gt = (int*) &gthr;
                    int *hv = (int*) &hashVector;
                    #pragma omp parallel for
                    for(int k=0;k<16;k++) {
                        if(iv[k]!=gt[k]) {
                            stackInput[k]=iv[k];
                            stackHash[k] = hv[k];
                            conPay[k] = ct[k];
                        }
                    }
                    //insert function does linear probing until keys are found or inserted
                    insert(stackInput,stackHash,conPay);
                }
            }
            else {
                    _mm512_mask_i32scatter_epi32 (&(hashtableKeys), m, hashVector, inputVector, 4);
                    _mm512_mask_i32scatter_epi32 (&(hashtablePay),m, hashVector, countVector, 4);
            }
            SIMDInsertEnd = clock();
            SIMDTime += ((long double)SIMDInsertEnd - (long double)SIMDInsertBegin)/CLOCKS_PER_SEC;

            continue;
        }

        m = _mm512_cmpeq_epi32_mask(inputVector, key_H);
        mask = _mm512_mask2int(m);

        if(mask!=65535) {
            SIMDInsertBegin = clock();
            __m512i payload = _mm512_i32gather_epi32 (hashVector, &(hashtablePay), 4);
            __m512i incr =  _mm512_mask_add_epi32 (incr, m, payload, countVector);
            _mm512_mask_i32scatter_epi32 (&(hashtablePay), m, hashVector, incr, 4);

            //check for empty slots in hashtable
            __mmask16  maskzero = _mm512_cmpeq_epi32_mask(key_H, zero);
            __m512i conf = _mm512_mask_conflict_epi32(conf,maskzero,hashVector);
            __mmask16 af = _mm512_cmpeq_epi32_mask(conf,zero);
            __mmask16 a = _mm512_mask_cmpeq_epi32_mask(maskzero,conf,zero);
            __mmask16 nf = _mm512_knot (af);
                    //if there are conflict in keys and hash index add to stack and do linear probing
            if((int)af!=65535){
                _mm512_mask_i32scatter_epi32 (&(hashtableKeys),a, hashVector,inputVector, 4);
                _mm512_mask_i32scatter_epi32 (&(hashtablePay),a,hashVector,countVector, 4);
                __m512i gthr = _mm512_i32gather_epi32(hashVector, &(hashtableKeys), 4);
                __mmask16 a2 = _mm512_cmpeq_epi32_mask(gthr,inputVector);
                int stackIp[16]= {0};
                int stackHs[16] = {0};
                int conP[16] = {0};
                if((int)a2!=65535) {
                    int stackIp[16]= {0};
                    int stackHs[16] = {0};
                    int conP[16] = {0};
                    int *iv = (int*) &inputVector;
                    int *ct = (int*) &countVector;
                    int *gt = (int*) &gthr;
                    int *hv = (int*) &hashVector;
                    #pragma omp parallel for
                    for(int k=0;k<16;k++) {
                        if(iv[k]!=gt[k]) {
                            stackIp[k]=iv[k];
                            stackHs[k] = hv[k];
                            conP[k] = ct[k];
                        }
                    }
                    insert(stackIp,stackHs,conP);
                }
            }

            if((int)af==65535)  {
                //if no conflict simply scatter
                _mm512_mask_i32scatter_epi32 (&(hashtableKeys), maskzero, hashVector, inputVector, 4);
                _mm512_mask_i32scatter_epi32 (&(hashtablePay), maskzero, hashVector, countVector, 4);
            }

            //gather again and check if there are still keys to be inserted and add to stack
            __m512i gthrA = _mm512_i32gather_epi32(hashVector, &(hashtableKeys), 4);
            __mmask16  maskconf = _mm512_cmpneq_epi32_mask(gthrA, inputVector);
            int stackI[16]  __attribute__ ((aligned (64))) = {0} , top=0;
            int stackH[16] __attribute__ ((aligned (64))) = {0};
            int stackB[16] __attribute__ ((aligned (64))) = {0};

            _mm512_mask_store_epi32 (&stackI[top], maskconf, inputVector);
            _mm512_mask_store_epi32 (&stackH[top], maskconf, hashVector);
            _mm512_mask_store_epi32 (&stackB[top], maskconf, countVector);
            insert(stackI,stackH,stackB);
            SIMDInsertEnd = clock();
            SIMDTime += ((long double)SIMDInsertEnd - (long double)SIMDInsertBegin)/CLOCKS_PER_SEC;
            continue;
        }
    }
    free(input);
    free(count);
}

void clearHash(){
int j;
  for(j=0;j<HSIZE;j++){
      hashtableKeys[j] = 0;
      hashtablePay[j] = 0;
    }
}

int hashCheck(){
   int count = 0;
int j;
   for(j=0;j<HSIZE;j++){
        if(hashtableKeys[j]!=0) {
            if(hashtablePay[j]!=0) {
               count++;
            }
      }
    }
  return count;
}

int addValues(){
    int count = 0;
    int j;
    for(j=0;j<HSIZE;j++){
        if(hashtablePay[j]!=0) {
            if(hashtableKeys[j]!=0) {
                count+=hashtablePay[j];
            }
        }
    }
    return count;
}


long double  getSIMDTime(){
  return SIMDTime;
}

void clearClocks(){
  SIMDTime=0;
}
