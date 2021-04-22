#include <immintrin.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <stdbool.h>
#include "CuckooVertical.h"

#define KEYSPERPAGE 1 
#define REPEATS 2 
#define HSIZE 5000000

clock_t SIMDInsertBegin, SIMDInsertEnd;

static long double SIMDTime;

int hashtableKeys1[HSIZE];
int hashtablePay1[HSIZE];
int hashtableKeys2[HSIZE];
int hashtablePay2[HSIZE];

unsigned int hash1(int key){
  return ((unsigned long)((unsigned int)1300000077*key)* HSIZE)>>32;
}

unsigned int hash2(int key){
  return ((unsigned long)((unsigned int)1145678917*key)* HSIZE)>>32;
}


void print128_num(__m512i var){
    int *resi = (int*) &var;
    std::cout << resi[0] << resi[1] << resi[2] << resi[3] << resi[4] << resi[5] << resi[6]
    << resi[7] << resi[8] << resi[9] << resi[10] << resi[11] << resi[12] << resi[13] << resi[14] << resi[15];
}

void insert(int arr[], int hv[],int cnt[]) {
 	__mmask16 m,j,g;
    int i=0,mask;
    __m512i hashVector2;

    __m512i inputVector = _mm512_setr_epi32 (arr[i], arr[i+1], arr[i+2], arr[i+3], arr[i+4], arr[i+5], arr[i+6],arr[i+7], arr[i+8], arr[i+9], arr[i+10], arr[i+11], arr[i+12], arr[i+13],arr[i+14],arr[i+15]);
	__m512i hashVector = _mm512_setr_epi32(hv[i], hv[i+1], hv[i+2], hv[i+3], hv[i+4], hv[i+5],hv[i+6], hv[i+7],hv[i+8], hv[i+9], hv[i+10], hv[i+11], hv[i+12], hv[i+13],hv[i+14],hv[i+15]);
    __m512i count = _mm512_setr_epi32(cnt[i],cnt[i+1], cnt[i+2], cnt[i+3], cnt[i+4], cnt[i+5], cnt[i+6], cnt[i+7],cnt[i+8], cnt[i+9], cnt[i+10], cnt[i+11], cnt[i+12], cnt[i+13], cnt[i+14], cnt[i+15]);
    __m512i zero = _mm512_setzero_epi32();
    __m512i setone = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);

    do {

        //gather keys and payloads from hashtable1 and scatter input vector to hashtable1 
        __mmask16 a = _mm512_cmpneq_epi32_mask(hashVector,zero);
        __m512i gthr1 = _mm512_i32gather_epi32(hashVector, &(hashtableKeys1), 4);
        __m512i payload1 = _mm512_i32gather_epi32(hashVector, &(hashtablePay1), 4);
        _mm512_mask_i32scatter_epi32 (&(hashtableKeys1),a,hashVector, inputVector, 4);
        _mm512_mask_i32scatter_epi32 (&(hashtablePay1),a,hashVector, count, 4);
        int *gt = (int*) &gthr1;
        int *ct = (int*) &count;
        int *ct1 = (int*) &payload1;
        int *hv2 = (int*) &hashVector2;
        int *iv = (int*) &inputVector;
        for(int k=0;k<16;k++) {
            iv[k] = gt[k];
            ct[k] = ct1[k];
        }
        for(int k=0;k<16;k++) {
            hv2[k] = hash2(iv[k]); //hash the swapped keys to 2nd table
        }

        //gather keys and payloads from hashtable2 
        __m512i gthr2 = _mm512_i32gather_epi32(hashVector2, &(hashtableKeys2), 4);
        __m512i payload2 = _mm512_i32gather_epi32(hashVector2, &(hashtablePay2), 4);
        //if the keys of hashtable2 matches with input vector increment payloads
        __mmask16 m1 = _mm512_mask_cmpeq_epi32_mask(a,inputVector, gthr2);
        __m512i incr = _mm512_add_epi32 (count, payload2);
        _mm512_mask_i32scatter_epi32 (&(hashtablePay2),m1,hashVector2,incr, 4);
        //set zero in input vector when the key is found or inserted
        inputVector = _mm512_mask_set1_epi32(inputVector,m1,0);

        //if the keys of hashtable2 has zeros insert input vector keys
        __mmask16 m2 = _mm512_mask_cmpeq_epi32_mask(a,gthr2,zero);
        _mm512_mask_i32scatter_epi32 (&(hashtableKeys2),m2,hashVector2,inputVector, 4);
        _mm512_mask_i32scatter_epi32 (&(hashtablePay2),m2,hashVector2,count, 4);
		inputVector = _mm512_mask_set1_epi32(inputVector,m2,0);

        j =  _mm512_cmpeq_epi32_mask(zero, inputVector);

        mask = (int) j;
        if(mask==65535) //exit loop when all input vector keys are inserted
            break;

        //repeat swap with hashtable1 if there was no match or empty slot in hashtable2
        __mmask16 n = _mm512_knot(j);
        _mm512_mask_i32scatter_epi32 (&(hashtableKeys2),n,hashVector2, inputVector, 4);
        _mm512_mask_i32scatter_epi32 (&(hashtablePay2),n,hashVector2, count, 4);
        int *gt2 = (int*) &gthr2;
        int *ct2 = (int*) &payload2;
        int *hv1 = (int*) &hashVector;
        int *iv2 = (int*) &inputVector;
        for(int k=0;k<16;k++) {
            inputVector = _mm512_mask_set1_epi32(inputVector,n,gt2[k]);
            count = _mm512_mask_set1_epi32(inputVector,n,ct2[k]);
        }
        for(int k=0;k<16;k++) {
            hv[k] = hash1(iv2[k]);
        }
        __m512i gthr = _mm512_i32gather_epi32(hashVector, &(hashtableKeys1), 4);
        __m512i pay = _mm512_i32gather_epi32(hashVector, &(hashtablePay1), 4);
        __mmask16 m3 = _mm512_mask_cmpeq_epi32_mask(n,inputVector, gthr);
        __m512i incr1 = _mm512_mask_add_epi32 (incr1,n,count, pay);
        _mm512_mask_i32scatter_epi32 (&(hashtablePay1),m,hashVector,incr1, 4);
        inputVector = _mm512_mask_set1_epi32(inputVector,m3,0);
        __mmask16 m4 = _mm512_mask_cmpeq_epi32_mask(n,gthr,zero);
        _mm512_mask_i32scatter_epi32 (&(hashtableKeys1),m4,hashVector,inputVector, 4);
        _mm512_mask_i32scatter_epi32 (&(hashtablePay1),m4,hashVector,count, 4);
        inputVector = _mm512_mask_set1_epi32(inputVector,m4,0);

        g =  _mm512_cmpeq_epi32_mask(zero, inputVector);

        mask = (int) g;
        if(mask==65535) //exit loop when all input vector keys are inserted
            break;
    } while(1);
}

int countsetbits(int n)
{
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}


int partition (int arr[], int low, int high)
{
    int pivot = arr[high];    
    int i = (low - 1);  
   #pragma omp parallel for
    for (int j = low; j <= high - 1; j++)
    {
         
        if (arr[j] < pivot)
        {
            i++;    
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
		int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}



int VectorProbeInsert(int arr[], int datasize,int c) {
    __mmask16  m1,m2,m10,m20;
    int mask1,mask2,mask10,mask20;
    __m512i inputVector,hashVector1, hashVector2;
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

        inputVector = _mm512_setr_epi32 (input[i], input[i+1], input[i+2], input[i+3], input[i+4], input[i+5], input[i+6],input[i+7], input[i+8], input[i+9], input[i+10], input[i+11], input[i+12], input[i+13],input[i+14],input[i+15]);
        int *hv1 = (int*) &hashVector1;
        int *hv2 = (int*) &hashVector2;
        int *iv = (int*) &inputVector;

        for(int k=0;k<16;k++) {
            hv1[k] = hash1(iv[k]);
            hv2[k] = hash2(iv[k]);
        }

        __m512i countVector =  _mm512_setr_epi32 (count[i], count[i+1],count[i+2], count[i+3],count[i+4], count[i+5], count[i+6],count[i+7], count[i+8], count[i+9],count[i+10], count[i+11], count[i+12], count[i+13],count[i+14],count[i+15]);

        //gather from hashtable1
		__m512i  key_H1 = _mm512_i32gather_epi32(hashVector1, &(hashtableKeys1), 4);

        //gather from hashtable2
        __m512i  key_H2 = _mm512_i32gather_epi32(hashVector2, &(hashtableKeys2), 4);

        //compare input keys with gathered keys
        m1 = _mm512_cmpeq_epi32_mask(inputVector, key_H1);
        m2 = _mm512_cmpeq_epi32_mask(inputVector, key_H2);

        mask1 = _mm512_mask2int(m1);
        mask2 = _mm512_mask2int(m2);

        //if keys are found
        if(mask1 == 65535 || mask2==65535) {

            //increment payloads
           // SIMDInsertBegin = clock();
            //probe in hashtabletable1 and increment payloads
            __m512i payload1 = _mm512_i32gather_epi32 (hashVector1, &(hashtablePay1), 4);
                __m512i incr = _mm512_add_epi32 (countVector, payload1);
                _mm512_mask_i32scatter_epi32 (&(hashtablePay1),mask1, hashVector1,incr, 4);

            //probe in hashtabletable2 and increment payloads
            __m512i payload2 = _mm512_i32gather_epi32 (hashVector2, &(hashtablePay2), 4);
                __m512i incri = _mm512_add_epi32 (countVector, payload2);
                _mm512_mask_i32scatter_epi32 (&(hashtablePay2),mask2, hashVector2,incri, 4);
            //SIMDInsertEnd = clock();
            //SIMDTime = SIMDTime +  ((long double)SIMDInsertEnd - (long double)SIMDInsertBegin)/CLOCKS_PER_SEC;
            continue;
        }

        //if there are empty slots in hashtable1 perform insertion
        m10 = _mm512_cmpeq_epi32_mask(key_H1, zero);
        m20 = _mm512_cmpeq_epi32_mask(key_H2, zero);
        mask10 = _mm512_mask2int(m10);
        mask20 = _mm512_mask2int(m20);

        if(mask10 == 65535) {
            SIMDInsertBegin = clock();
            int cnt[16] = {0};
                int conPay[16] __attribute__ ((aligned (64))) = {0};
                //check conflict in input vector
				__m512i con1 = _mm512_conflict_epi32(hashVector1);
                __mmask16 a1 = _mm512_cmpeq_epi32_mask(con1,zero);
                if((int)a1!=65535){
                        //first scatter non-conflicting keys
                    _mm512_mask_i32scatter_epi32 (&(hashtableKeys1),a1,hashVector1,inputVector, 4);
                    _mm512_mask_i32scatter_epi32 (&(hashtablePay1),a1, hashVector1,countVector, 4);

                    //gather keys again, if still some keys are not inserted add to stack for swapping 
                    __m512i gthr = _mm512_i32gather_epi32(hashVector1, &(hashtableKeys1), 4);
                    __mmask16 a2 = _mm512_cmpeq_epi32_mask(gthr,inputVector);
                    if((int)a2!=65535) {
                        int stackInput[16]= {0};
                        int stackHash[16] = {0};
                        int *iv = (int*) &inputVector;
                        int *ct = (int*) &countVector;
                        int *gt = (int*) &gthr;
                        int *hv = (int*) &hashVector1;
                        for(int k=0;k<16;k++) {
                           if(iv[k]!=gt[k]) {
                               stackInput[k]=iv[k];
                               stackHash[k] = hv[k];
                               conPay[k] = ct[k];
                            }
                        }
                        //call insertion method for swapping input keys
                        insert(stackInput,stackHash,conPay);
                    }
                }

                //no conflicting keys
                //check for conflicting hash before scatter
                else {
                        //if no conflicting keys or hash scatter safely all input keys to hashtable1
                    _mm512_mask_i32scatter_epi32 (&(hashtableKeys1), m10, hashVector1, inputVector, 4);
                    _mm512_mask_i32scatter_epi32 (&(hashtablePay1),m10, hashVector1, countVector, 4);
                }
            SIMDInsertEnd = clock();
            SIMDTime += ((long double)SIMDInsertEnd - (long double)SIMDInsertBegin)/CLOCKS_PER_SEC;
            continue;
        }

        //if only some keys are equal to gathered keys or contains zero or has collisions 
		if(mask10!=65535) {
            SIMDInsertBegin = clock();

            //check for matching keys in hashtable1 and increment payloads
            __mmask16 cmp2 = _mm512_cmpeq_epi32_mask(key_H1,inputVector);
            __mmask16 ne =_mm512_knot (cmp2);
            __m512i payload = _mm512_i32gather_epi32 (hashVector1, &(hashtablePay1), 4);
            __m512i incr = _mm512_mask_add_epi32 (incr,cmp2,payload, countVector);
            _mm512_mask_i32scatter_epi32 (&(hashtablePay1),cmp2,hashVector1,incr,4);
            inputVector = _mm512_mask_set1_epi32(inputVector,cmp2,0);

            __mmask16 cmp1 = _mm512_cmpeq_epi32_mask(key_H1, zero);
            //check if the keys are present in hashtable2 locations
            __mmask16 cmpk2 = _mm512_mask_cmpeq_epi32_mask(ne,key_H2,inputVector);
            __m512i payk2 = _mm512_i32gather_epi32 (hashVector2, &(hashtablePay2), 4);
            __m512i incrk2 = _mm512_mask_add_epi32 (incrk2,cmpk2,payk2, countVector);
            _mm512_mask_i32scatter_epi32 (&(hashtablePay2),cmpk2,hashVector2,incrk2,4);
            inputVector = _mm512_mask_set1_epi32(inputVector,cmpk2,0);

            //insert keys if hashtable1 locations are empty
            __m512i gt1 = _mm512_i32gather_epi32 (hashVector1, &(hashtableKeys1), 4);
            __mmask16 cmp3 = _mm512_cmpeq_epi32_mask(gt1, zero);

            //check for conflicts in key and hash values before scatter
            __m512i con1 = _mm512_mask_conflict_epi32(con1,cmp3,hashVector1);
            __mmask16 a1 = _mm512_mask_cmpeq_epi32_mask(cmp3,con1,zero);
            if((int)a1!=65535) {
 			//scatter non-conflicting keys first
                _mm512_mask_i32scatter_epi32 (&(hashtableKeys1),a1,hashVector1,inputVector, 4);
                _mm512_mask_i32scatter_epi32 (&(hashtablePay1),a1, hashVector1,countVector, 4);
                inputVector = _mm512_mask_set1_epi32(inputVector,a1,0);
                __m512i gthr = _mm512_i32gather_epi32(hashVector1, &(hashtableKeys1), 4);
                __mmask16 a2 = _mm512_cmpeq_epi32_mask(gthr,inputVector);
                //if still keys are not inserted add to stack for swapping
                if((int)a2!=65535) {
                    int stackInput[16]= {0};
                    int stackHash[16] = {0};
                    int conP[16] = {0};
                    int *iv = (int*) &inputVector;
                    int *ct = (int*) &countVector;
                    int *gt = (int*) &gthr;
                    int *hv = (int*) &hashVector1;
					for(int k=0;k<16;k++) {
                        if(iv[k]!=gt[k] && iv[k]!=0) {
                            stackInput[k]=iv[k];
                            stackHash[k] = hv[k];
                            conP[k] = ct[k];
                            iv[k] = 0;
                        }
                    }
                    //dropduplicates(stackInput,stackHash,conP);
                    insert(stackInput,stackHash,conP);
                }
            }
            else if(((int)a1!=65535)){
                   //scatter to hashtable1 if no conflict 
               _mm512_mask_i32scatter_epi32 (&(hashtableKeys1),cmp3,hashVector1,inputVector, 4);
               _mm512_mask_i32scatter_epi32 (&(hashtablePay1),cmp3,hashVector1,countVector, 4);
               inputVector = _mm512_mask_set1_epi32(inputVector,cmp3,0);
            }


            //if still keys remaining to be inserted put in stack 
            __mmask16 cmp5 = _mm512_cmpeq_epi32_mask(inputVector, zero);
            if(cmp5!=65535) {
                int stackI[16] __attribute__ ((aligned (64))) = {0};
                int stackH[16] __attribute__ ((aligned (64))) = {0};
                int stackB[16] __attribute__ ((aligned (64))) = {0};
                _mm512_mask_store_epi32 (&stackI, cmp5, inputVector);
                _mm512_mask_store_epi32 (&stackH, cmp5, hashVector1);
                _mm512_mask_store_epi32 (&stackB, cmp5, countVector);
                //remove duplicates in stack
	            //dropduplicates(stackI,stackH,stackB);
                insert(stackI,stackH,stackB);
            }
            SIMDInsertEnd = clock();
            SIMDTime = SIMDTime +  ((long double)SIMDInsertEnd - (long double)SIMDInsertBegin)/CLOCKS_PER_SEC;
            continue;
        }
    }
    free(input);
    free(count);
}

void clearHash(){
   int j;
    for(j=0;j<HSIZE;j++){
      hashtableKeys1[j] = 0;
      hashtablePay1[j] = 0;
      hashtableKeys2[j] = 0;
      hashtablePay2[j] = 0;

    }
}


int hashCheck(){
    int count1 = 0;
    for(int j=0;j<HSIZE;j++){
        if((hashtableKeys1[j]!=0)) {
           count1++;
        }
    }
    int count2 = 0;
    for(int j=0;j<HSIZE;j++){
        if((hashtableKeys2[j]!=0)) {
            count2++;
        }
    }
    int count = count1+count2;
    return count;
}
int addValues(){
    int count1 = 0;

    for(int j=0;j<HSIZE;j++){
        if(hashtableKeys1[j]!=0) {
            if(hashtablePay1[j]!=0) {
                count1+=hashtablePay1[j];
            }
        }
    }
    int count2 = 0;
    for(int j=0;j<HSIZE;j++){
        if(hashtableKeys2[j]!=0) {
           if(hashtablePay2[j]!=0) {
count2+=hashtablePay2[j];
            }
        }
    }
    int count = count1+count2;
    return count;
}

long double  getSIMDTime(){
  return SIMDTime;
}

void clearClocks(){
  SIMDTime=0;
}
