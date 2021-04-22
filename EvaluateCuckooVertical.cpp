#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include "distribution.h"
#include "CuckooVertical.h"

using namespace std;

__attribute__((optimize("no-tree-vectorize")))

int  main(int argc, char** argv){

    int dataSize;
    int insertValue;
    int initialSize = 500000;
    int totalSize = 5000000;
    int iteration;
/*
cuckoo Hashing Vertical Vectorization Evaluation
*/

    for(iteration=0;iteration<20;iteration++){
        cout << "\nVertical vectorization cuckoo hashing\n";

        short DistributionType=0;     
        while(DistributionType<4){
            switch(DistributionType){

                case 0 :    cout << "\nDense Unique Random Values\n";
                                break;
                case 1 :    cout << "\nSequential Values\n";
                                break;
                case 2 :    cout << "\nUniform Random Values\n";
                                break;
                case 3 :    cout << "\nExponential Values\n";
                                break;
            }

            cout << "Datasize\tunique\ttotal\tVV time\n";
            for(dataSize=initialSize;dataSize<=totalSize;dataSize+=initialSize){
                int i,j,c;
                int *arr = (int *)calloc(dataSize,sizeof(int));
                setGen(dataSize); //Initialize generation Size

                switch(DistributionType){ //Populate with values

                    case 0:
                        setGen(dataSize*3);
                        InitDenseUnique();
                        for(i=0;i<dataSize;i+=1){
                            arr[i] = DenseUniqueRandom();
                        }
                        c=0;
                        VectorProbeInsert(arr,dataSize,c);
                        break;

                    case 1:
                        for(i=0;i<dataSize;i++){
                            arr[i] = SequentialNumbers(5);
                            }
                        c=1;
                        VectorProbeInsert(arr,dataSize,c);
                        break;

                    case 2:
                        for(i=0;i<dataSize;i++){
                                    arr[i] = UniformRandom();
                        }
                        c=2;
                        VectorProbeInsert(arr,dataSize,c);
                        break;

                    case 3:
                        for(i=0;i<dataSize;i+=1){
                                arr[i] = Exponential();
                            }
                        c=3;
                        VectorProbeInsert(arr,dataSize,c);
                        break;
                }
                cout << dataSize << "\t" << hashCheck() << "\t" << addValues() << "\t" << getSIMDTime() << "\n";
                clearHash();
                clearClocks();
            }

            DistributionType++;
        }
    }
}
