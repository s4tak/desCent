# desCent
A simple and intuitive library to create and train MLP neuronal networks.


##Training & Testing Example

```
#include <stdio.h>
#include "ml.h"
#include "algebra2.h"


int main(){
    int layers[]={774, 16, 16, 1};
    short nLayers=sizeof(layers)/sizeof(layers)[0];

    char csvPath[]="/home/passpow/chessIA/datasets/test/finaldb.csv";
    char outPath[]="/home/passpow/chessIA/spasskyFiles/";
    int r,c;
    float *dataSet;
    Layer* spassky = createNetwork(nLayers, layers);
    printNetwork(spassky);
    printf("[*] Leyendo dataset\n");
    if(importMatrixCsv(&r, &c, &dataSet, csvPath)!=0){
        printf("ERROR READING THE DATASET");
        return -1;
    }
    printf("[*] Dataset completado\n");
    importNetwork(spassky, outPath);
    for (int i=10;i<=10000; i*=10){
        printf("Training network with alfa %f\n", 1.f/i);
        trainNetwork(spassky, dataSet, 1.f/i, 1, 774, 0, 1, r);
    }

    exportNetwork(spassky, outPath);
    
    return 0;
}

```
