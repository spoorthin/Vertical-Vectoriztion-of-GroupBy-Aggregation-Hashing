#include <immintrin.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <stdbool.h>
#include "2ChoiceVertical.h"

#define KEYSPERPAGE 1
#define REPEATS 2
#define HSIZE 5000000

clock_t SIMDInsertBegin, SIMDInsertEnd;

static long double SIMDTime;

int hashtableKeys[HSIZE];
int hashtablePay[HSIZE];


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

void insert(int arr[], int hv[],int cnt[] ) {
    //This function inserts keys that were invalidated from input lanes from hashvector1 location
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

             //check for any conflict in hash index and increment
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
            __m512i pay = _mm512_i32gather_epi32(hashVector, &(hashtablePay), 4);
            __m512i incr = _mm512_mask_add_epi32(incr,m1,pay,count);
            _mm512_mask_i32scatter_epi32 (&(hashtablePay),m1,hashVector, incr, 4);

            //set finished keys to zero
            inputVector = _mm512_mask_set1_epi32 (inputVector, m1, 0);
            hashVector = _mm512_mask_set1_epi32 (hashVector, m1, 0);
            
            //insert keys if there is empty slot in the hashtable
           __m512i key_HH = _mm512_i32gather_epi32(hashVector, &(hashtableKeys), 4);
            __mmask16 m2 = _mm512_cmpeq_epi32_mask(zero, key_HH);
           _mm512_mask_i32scatter_epi32 (&(hashtableKeys), m2,hashVector, inputVector, 4);
            _mm512_mask_i32scatter_epi32 (&(hashtablePay),m2,hashVector, count, 4);

            //set finished keys to zero
            inputVector = _mm512_mask_set1_epi32 (inputVector, m2, 0);
            hashVector = _mm512_mask_set1_epi32 (hashVector, m2, 0);
            
            //loop until all input vector keys are zero
            j =  _mm512_cmpeq_epi32_mask(zero, inputVector);
            mask = (int) j;
            if(mask==65535)
              break;

    } while(1);

}


int VectorProbeInsert(int arr[], int datasize,int c) {

    __mmask16  m1,m2,m10,m20;
    __m512i inputVector, hashVector1, hashVector2;
    __m512i setone = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
    int  mask1, mask2,mask10,mask20;
    int ecount = 0;
    int p=0;
    int j;

    __m512i zero = _mm512_setzero_epi32();

    int mask,cnt;
    int count[16] = {0};
    int input[16] = {0};

    #pragma omp parallel while
    while(p<datasize) {
        if(c==2||c==3){
            bool flag=false;
            for(j=0;j<16;j++) {
                 if(input[j]==arr[p]) {
                count[j]+=1;
                flag = true;
                break;
            }
         }

        if (flag==false) {
                for(int s=0;s<16;s++) {
                    if(input[s]==0){
                        input[s]=arr[p];
                        count[s]=1;
                        break;
                    }
                }
            }
        }

        if(c==0||c==1) {
            for(int k=0;k<16;k++) {
                if(input[k]==0){
                    input[k]=arr[p];
                     count[k]=1;
                    break;
                }
            }
        }

        if(input[15]!=0)      {
            int i=0;
            inputVector = _mm512_setr_epi32 (input[i], input[i+1], input[i+2], input[i+3], input[i+4], input[i+5], input[i+6],input[i+7], input[i+8], input[i+9], input[i+10], input[i+11], input[i+12], input[i+13],input[i+14],input[i+15]);

            __m512i  countVector =  _mm512_setr_epi32 (count[i], count[i+1],count[i+2], count[i+3],count[i+4], count[i+5], count[i+6],count[i+7], count[i+8], count[i+9],count[i+10], count[i+11], count[i+12], count[i+13],count[i+14],count[i+15]);

            //set hashvector1 and hashvector2 with two potential hash locations for each key in input vector
            int *hv1 = (int*) &hashVector1;
            int *hv2 = (int*) &hashVector2;
            int *iv = (int*) &inputVector;

            for(int k=0;k<16;k++) {
                hv1[k] = hash1(iv[k]);
                hv2[k] = hash2(iv[k]);
            }

            //gather keys from hash location1
            __m512i  key_H1 = _mm512_i32gather_epi32(hashVector1, &(hashtableKeys), 4);

            //gather keys from hash location2
            __m512i  key_H2 = _mm512_i32gather_epi32(hashVector2, &(hashtableKeys), 4);
            //compare input keys and gathered keys
            m1 = _mm512_cmpeq_epi32_mask(inputVector, key_H1);
            m2 = _mm512_cmpeq_epi32_mask(inputVector, key_H2);

            mask1 = _mm512_mask2int(m1);
            mask2 = _mm512_mask2int(m2);

            //check if gathered1 locations are empty
            m10 = _mm512_cmpeq_epi32_mask(key_H1, zero);
            m20 = _mm512_cmpeq_epi32_mask(key_H2, zero);
            mask10 = _mm512_mask2int(m10);
            mask20 = _mm512_mask2int(m20);

            //perform insertion if hash locations of hashvector1 has empty slots
            if(mask10 == 65535 ){
                 //similar insertion steps as linear probing
                 SIMDInsertBegin = clock();

                //check for conflicts in keys and hash values
                __m512i con1 = _mm512_conflict_epi32(hashVector1);
                __mmask16 a1 = _mm512_cmpeq_epi32_mask(con1,zero);
                    if((int)a1!=65535){
                        _mm512_mask_i32scatter_epi32 (&(hashtableKeys),a1,hashVector1,inputVector, 4);
                        _mm512_mask_i32scatter_epi32 (&(hashtablePay),a1, hashVector1,countVector, 4);
                        __m512i gthr = _mm512_i32gather_epi32(hashVector1, &(hashtableKeys), 4);
                        __mmask16 a2 = _mm512_cmpeq_epi32_mask(gthr,inputVector);
                        if((int)a2!=65535) {
                            int stackInput[16]= {0};
                            int stackHash[16] = {0};
                            int conPay[16] = {0};
                            int *iv = (int*) &inputVector;
                            int *gt = (int*) &gthr;
                            int *hv = (int*) &hashVector1;
                            int *ct = (int*) &countVector;
                            for(int k=0;k<16;k++) {
                                if(iv[k]!=gt[k]) {
                                    stackInput[k]=iv[k];
                                    stackHash[k] = hv[k];
                                    conPay[k] = ct[k];
                                }
                            }
                            insert(stackInput,stackHash,conPay);
                        }
                    }
                    else {
                        //if no conflicting keys and hash simply scatter all keys to hashvector1 locations
                        _mm512_mask_i32scatter_epi32 (&(hashtableKeys), m10, hashVector1, inputVector, 4);
                        _mm512_mask_i32scatter_epi32 (&(hashtablePay),m10, hashVector1, countVector, 4);
                    }
                SIMDInsertEnd = clock();
                SIMDTime += ((long double)SIMDInsertEnd - (long double)SIMDInsertBegin)/CLOCKS_PER_SEC;
            }

            //if input keys are matching with gathered from both hash locations perform aggregation
            if(mask1 == 65535 || mask2==65535) {
                //increment payloads
                SIMDInsertBegin = clock();

                //increment the payloads of hashvector1 locations with number of occurences if there is a match
                __m512i payload1 = _mm512_i32gather_epi32 (hashVector1, &(hashtablePay), 4);
                __m512i incr1 = _mm512_add_epi32 (payload1, countVector);
                _mm512_mask_i32scatter_epi32 (&(hashtablePay), mask1,hashVector1, incr1, 4);
                //increment the payloads of hashvector2 locations with number of occurences if there is a match
                __m512i payload2 = _mm512_i32gather_epi32 (hashVector2, &(hashtablePay), 4);
                __m512i incr2 = _mm512_add_epi32 (payload2, countVector);
                _mm512_mask_i32scatter_epi32 (&(hashtablePay), mask2,hashVector2, incr2, 4);
                SIMDInsertEnd = clock();
                SIMDTime = SIMDTime +  ((long double)SIMDInsertEnd - (long double)SIMDInsertBegin)/CLOCKS_PER_SEC;

            }

            //if there are only some matches and empty slots in hashvector1 locations,
            //choose hashvector2 locations to probe or insert
            if(mask1!=65535) {
                SIMDInsertBegin = clock();
                //increment payloads for matching input keys in gathered keys at hashvector1 location
                __mmask16 cmp2 = _mm512_cmpeq_epi32_mask(key_H1,inputVector);
                __m512i pay = _mm512_i32gather_epi32 (hashVector1, &(hashtablePay), 4);
                __m512i incr = _mm512_mask_add_epi32 (incr,cmp2,pay, countVector);
                            _mm512_mask_i32scatter_epi32 (&(hashtablePay),cmp2,hashVector1,incr,4);
                //set input vector to zero for finished keys
                inputVector = _mm512_mask_set1_epi32(inputVector,cmp2,0);

                //check if the keys are present in hashvector2 locations and increment payloads
                __mmask16 cmpk2 = _mm512_cmpeq_epi32_mask(key_H2,inputVector);
                __m512i payk2 = _mm512_i32gather_epi32 (hashVector2, &(hashtablePay), 4);
                __m512i incrk2 = _mm512_mask_add_epi32 (incrk2,cmpk2,payk2, countVector);
                _mm512_mask_i32scatter_epi32 (&(hashtablePay),cmpk2,hashVector2,incrk2,4);
                inputVector = _mm512_mask_set1_epi32(inputVector,cmpk2,0);

                //insert keys if hashvector1 locations are empty
                __m512i gt1 = _mm512_i32gather_epi32 (hashVector1, &(hashtableKeys), 4);
                __mmask16 cmp3 = _mm512_cmpeq_epi32_mask(gt1, zero);

                //check for conflicts
                __m512i con1 = _mm512_mask_conflict_epi32(con1,cmp3,hashVector1);
                __mmask16 a1 = _mm512_mask_cmpeq_epi32_mask(cmp3,con1,zero);
                if(((int)a1!=65535)) {
                    _mm512_mask_i32scatter_epi32 (&(hashtableKeys),a1,hashVector1,inputVector, 4);
                    _mm512_mask_i32scatter_epi32 (&(hashtablePay),a1, hashVector1,countVector, 4);
                    inputVector = _mm512_mask_set1_epi32(inputVector,a1,0);
                    __m512i gthr = _mm512_i32gather_epi32(hashVector1, &(hashtableKeys), 4);
                    __mmask16 a2 = _mm512_cmpeq_epi32_mask(gthr,inputVector);

                    //if collisions add to stack and linear probing is done
                    if((int)a2!=65535) {
                        int stackIp[16]= {0};
                        int stackHs[16] = {0};
                        int conP[16] = {0};
                        int *ivv = (int*) &inputVector;
                        int *gtt = (int*) &gthr;
                        int *hvv = (int*) &hashVector1;
                        int *ct = (int*) &countVector;
                        for(int k=0;k<16;k++) {
                            if(ivv[k]!=gtt[k] && ivv[k]!=0) {
                                stackIp[k]=ivv[k];
                                stackHs[k] = hvv[k];
                                conP[k] = ct[k];
                                ivv[k] = 0;
                            }
                        }
                        insert(stackIp,stackHs,conP);

                    }
                }

                else if((int)a1==65535){
                    _mm512_mask_i32scatter_epi32 (&(hashtableKeys),cmp3,hashVector1,inputVector, 4);
                    _mm512_mask_i32scatter_epi32 (&(hashtablePay),cmp3,hashVector1,countVector, 4);
                    inputVector = _mm512_mask_set1_epi32(inputVector,cmp3,0);
                }

                //check if there is empty location in hashvector2 for keys which were not
                //inserted in hashvector1 positions
                __mmask16 cmp4 = _mm512_cmpneq_epi32_mask(inputVector, zero);
                __m512i gt2 = _mm512_i32gather_epi32 (hashVector2, &(hashtableKeys), 4);
                __mmask16 cmpk3 = _mm512_mask_cmpeq_epi32_mask(cmp4,gt2, zero);
                __m512i conh = _mm512_mask_conflict_epi32(conh,cmpk3,hashVector2);
                __mmask16 ah = _mm512_mask_cmpeq_epi32_mask(cmpk3,conh,zero);
                if((int)ah!=65535) {
                    _mm512_mask_i32scatter_epi32 (&(hashtableKeys),ah,hashVector2,inputVector, 4);
                    _mm512_mask_i32scatter_epi32 (&(hashtablePay),ah, hashVector2,countVector, 4);
                    inputVector = _mm512_mask_set1_epi32(inputVector,ah,0);
                    __m512i gthr = _mm512_i32gather_epi32(hashVector2, &(hashtableKeys), 4);
                    __mmask16 a2 = _mm512_cmpeq_epi32_mask(gthr,inputVector);

                    if((int)a2!=65535) {
                        int stackIp[16]= {0};
                        int stackHs[16] = {0};
                        int conP[16] = {0};
                        int *ivv = (int*) &inputVector;
                        int *gtt = (int*) &gthr;
                        int *hvv = (int*) &hashVector2;
                        int *ct = (int*) &countVector;
                        for(int k=0;k<16;k++) {
                            if(ivv[k]!=gtt[k] && ivv[k]!=0) {
                                stackIp[k]=ivv[k];
                                stackHs[k] = hvv[k];
                                conP[k] = ct[k];
                            }
                        }
                        insert(stackIp,stackHs,conP);
                    }
                }

                else if((int)ah==65535) {
                    _mm512_mask_i32scatter_epi32 (&(hashtableKeys),cmpk3,hashVector2,inputVector, 4);
                    _mm512_mask_i32scatter_epi32 (&(hashtablePay),cmpk3,hashVector2,countVector, 4);
                    inputVector = _mm512_mask_set1_epi32(inputVector,cmpk3,0);
                }

                //if still keys remaining to be inserted put in stack and linear probe from hash1 location
                __mmask16 cmp5 = _mm512_cmpneq_epi32_mask(inputVector, zero);
                int stackI[16] __attribute__ ((aligned (64))) = {0};
                int stackH[16] __attribute__ ((aligned (64))) = {0};
                int stackB[16] __attribute__ ((aligned (64))) = {0};
                _mm512_mask_store_epi32 (&stackI, cmp5, inputVector);
                _mm512_mask_store_epi32 (&stackH, cmp5, hashVector1);
                _mm512_mask_store_epi32 (&stackB, cmp5, countVector);
                //remove duplicates in stack
                // dropduplicates(stackI,stackH,stackB);
                insert(stackI,stackH,stackB);
                SIMDInsertEnd = clock();
                SIMDTime = SIMDTime +  ((long double)SIMDInsertEnd - (long double)SIMDInsertBegin)/CLOCKS_PER_SEC;
            }
            for(int s=0;s<16;s++) {
                    count[s] = 0;
                    input[s] = 0;
            }
        }
        p=p+1;
    }
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