#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include "ml.h"
#include "algebra.h"

int countLayers(Layer *lastLayer){
    int total=0;
    Layer *actualLayer=lastLayer;
    while((actualLayer->prevLayer)!=NULL){
        total++;
        actualLayer=actualLayer->prevLayer;
    }
    return total;
}

void computeOutput(Layer *firstLayer){
    Layer *actualLayer = firstLayer;
    Layer *prevLayer   = NULL;

    while(actualLayer!=NULL){

        if(prevLayer==NULL){
            prevLayer=actualLayer;
            actualLayer=actualLayer->nextLayer;
            continue;
        }

        int rows=actualLayer->neurons;
        int cols=actualLayer->inputSize;

        multMatrix(rows, cols,
                   actualLayer->weights,
                   cols, 1,
                   prevLayer->outputActivation,
                   actualLayer->output);

        addMatrix(rows,1,
                  actualLayer->output,
                  actualLayer->bias,
                  actualLayer->output);

        sigmoidMatrix(rows,1,
                      (float (*)[1])actualLayer->output,
                      (float (*)[1])actualLayer->outputActivation);

        prevLayer = actualLayer;
        actualLayer = actualLayer->nextLayer;
    }
    return;
}
int setInput(float *input, int inputSize, Layer *firstLayer){
    if ((firstLayer->neurons)!=inputSize){
        printf("%d, %d\n", firstLayer->neurons, inputSize);
        printf("The lenght of the array differs with the input of the layer!\n");
        return -1;
    }
    for (int i=0; i<firstLayer->neurons;i++){
        firstLayer->outputActivation[i]=input[i];
    }
    return 0;
}

int readDataset(char *csvPath, int numLabels, int numFeatures, int numRow, char ***arr){

    FILE *f;
    char buf[65536]; // Modify for bigger input IDK :>
    int current=0;

    f=fopen(csvPath, "r");
    if (f==NULL){
        printf("ERROR READING THE FILE %s", csvPath);
        return -1;
    }


    while(fgets(buf, sizeof(buf), f)){     
        printf("%s \n", buf);
        if (current==0) continue;  
        scanf("", &current);
        if (current==numRow){
            int c = split(buf, ',', arr);
            fclose(f);
            if (c!=(numLabels+numFeatures)){
                printf("DATASET POSIBLEMENTE DAÑADO\n");
                printf("números por línea: %d, numLabels+num:%d\n", c, numFeatures+numLabels);
                return -1; 
            }
            return c;
        }
        current++;
    } 
    fclose(f);
    return -1;
   
}

void printLayer(Layer *actualLayer){
    printf("---------------LAYER [%d]---------------\n", actualLayer->n);
    printf("LAYER Neurons:   %d\n", actualLayer->neurons);
    printf("LAYER InputSize: %d\n", actualLayer->inputSize); 
    printf("LAYER actualLayer: %p\n", actualLayer); 
    printf("LAYER nextLayer: %p\n", actualLayer->nextLayer); 
    printf("LAYER prevLayer: %p\n", actualLayer->prevLayer); 
}

float testNetwork(Layer *lastLayer, float *dataSet, int numLabels, int numFeatures, float min, float max, int datasetLines){
    Layer *firstLayer=lastLayer;
    float yExpected[numLabels][1]; 
    float result=0;
    //GET FIRST LAYER
    while (firstLayer->prevLayer!=NULL){
        firstLayer=firstLayer->prevLayer;
    }
    

    for (int line=0; line<datasetLines; line++){

        for (int i = 0; i < numLabels; i++){
            yExpected[i][0]=dataSet[line*(numLabels+numFeatures) + i];
        }

        for (int i = numLabels; i < numFeatures+numLabels; i++){
            (firstLayer->outputActivation)[i-numLabels]=(dataSet[line*(numLabels+numFeatures) + i]-min)/(max-min);
        }
        
        computeOutput(firstLayer);
        printf("%f, %f\n", yExpected[0][0], lastLayer->outputActivation[0]);
        //costFunction(numLabels, (float (*)[numLabels])lastLayer->outputActivation ,yExpected);
        for (int row=0; row<numLabels; row++){
            result+=powf(lastLayer->outputActivation[row]-yExpected[row][0], 2);
        }

       

    }
    return result/2;
}
int trainNetwork(Layer *lastLayer, float *dataSet, float learningRate, int numLabels, int numFeatures, float min, float max, int datasetLines){
    // USES BACKPROP WITH SGD TO CHANGE THE WEIGHTS AND BIASES
    Layer *firstLayer=lastLayer;
    float yExpected[numLabels][1]; 

    //GET FIRST LAYER
    while (firstLayer->prevLayer!=NULL){
        firstLayer=firstLayer->prevLayer;
    }
    

    for (int line=0; line<datasetLines; line++){

        //READ DATASET
        for (int i = 0; i < numLabels; i++){
            yExpected[i][0]=dataSet[line*(numLabels+numFeatures) + i];
        }
        //printf("%d \n", firstLayer->neurons);
        for (int i = numLabels; i < numFeatures+numLabels; i++){
            (firstLayer->outputActivation)[i-numLabels]=(dataSet[line*(numLabels+numFeatures)+i] - min)/(max-min);
        }
        
        computeOutput(firstLayer);
        //START BACKPROPPP

        Layer *actualLayer=lastLayer;
        while(actualLayer->prevLayer!=NULL){
            //ULTIMA CAPA
            if (actualLayer->nextLayer==NULL){ 
                scalarMatrixMult(numLabels, 1, yExpected, -1, yExpected);
                addMatrix(actualLayer->neurons, 1, actualLayer->outputActivation, (float *)yExpected, actualLayer->delta);

                //SOBREESCRIBO ZL pq no se vuelve a usar
                sigmoidDerivateMatrix(actualLayer->neurons, 1, actualLayer->output, actualLayer->output);
                hadamard(
                    actualLayer->neurons,
                    1,
                    actualLayer->delta,
                    actualLayer->output,
                    actualLayer->delta);
                actualLayer=(actualLayer->prevLayer);
                continue;
            }
            //RESTO DE CAPAS
            transpose(
                    actualLayer->nextLayer->neurons,
                    actualLayer->neurons,
                    actualLayer->nextLayer->weights,
                    actualLayer->nextLayer->weightsT
            );
            multMatrix(
                actualLayer->neurons,
                actualLayer->nextLayer->neurons,
                actualLayer->nextLayer->weightsT,
                (actualLayer->nextLayer)->neurons,
                1,
                actualLayer->nextLayer->delta,
                actualLayer->wTd
            );
            sigmoidDerivateMatrix(actualLayer->neurons, 1, actualLayer->output, actualLayer->output); //SOBREESCRIBO ZL
            hadamard(
                actualLayer->neurons, 
                1,
                actualLayer->wTd,
                actualLayer->output, 
                actualLayer->delta
            );
            actualLayer=(actualLayer->prevLayer);
 
        }
        

        /*
        for (int i = 0; i < actualLayer->neurons; i++) {
            float d = actualLayer->delta[i];
            for (int j = 0; j < actualLayer->inputSize; j++) {
                (float (*)[actualLayer->neurons])(actualLayer->weights)[i][j] -= learningRate *d * actualLayer->prevLayer->outputActivation[j];
            }
        }

        for (int i = 0; i < actualLayer->neurons; i++) {
            actualLayer->bias[i] -= learningRate * actualLayer->delta[i];
        }
        */


        
        actualLayer=lastLayer;
        while (actualLayer->prevLayer!=NULL){
            /*
            if (isnan(actualLayer->output[0])) {
                printf("¡NAN detectado en Capa %d!\n", actualLayer->n);
                return 0;
            }
            
            for(int i=0;i<actualLayer->neurons;i++)
                if(isnan(actualLayer->outputActivation[i])){
                    printf("NaN en capa %d neurona %d\n",actualLayer->n,i);
                    exit(1);
            }
            */
            //PESOS
            multMatrix(
                actualLayer->neurons,
                1,
                actualLayer->delta,
                1,
                actualLayer->inputSize,
                actualLayer->prevLayer->outputActivation,
                actualLayer->weightsGrad
            );
            scalarMatrixMult(
                actualLayer->neurons,
                actualLayer->inputSize,
                (float (*)[actualLayer->inputSize])actualLayer->weightsGrad,
                -learningRate, 
                (float (*)[actualLayer->inputSize])actualLayer->weightsGrad
            );
            addMatrix(
                actualLayer->neurons,
                actualLayer->inputSize,
                actualLayer->weightsGrad,
                actualLayer->weights,
                actualLayer->weights
            );
            
            //BIAS
            scalarMatrixMult(
                actualLayer->neurons,
                1,
                (float (*)[actualLayer->neurons])actualLayer->delta, 
                -learningRate,
                (float(*)[actualLayer->neurons])actualLayer->delta
            );
            addMatrix(
                actualLayer->neurons,
                1,
                actualLayer->bias,
                actualLayer->delta,
                actualLayer->bias
            );
            actualLayer=actualLayer->prevLayer;
            
        }
    } 
    return 0;
}

Layer* getFirstLayer(Layer* someLayer){
    Layer *actualLayer;
    actualLayer=someLayer;
    while(actualLayer->prevLayer!=NULL){
        actualLayer=actualLayer->prevLayer;
    }
    return actualLayer;
}

Layer* getLastLayer(Layer* someLayer){
    Layer *actualLayer;
    actualLayer=someLayer;
    while(actualLayer->nextLayer!=NULL){
        actualLayer=actualLayer->nextLayer;
    }
    return actualLayer;
}

void createLayer(Layer *memDir, int n, int neurons, int inputSize, Layer *prevLayer){
    memDir->n                = n;
    memDir->neurons          = neurons;
    memDir->inputSize        = inputSize;
    memDir->outputActivation = malloc(sizeof(float)*neurons);
    memDir->prevLayer        = prevLayer;
    memDir->nextLayer        = NULL; 
    if (prevLayer==NULL) return; //INPUT LAYER
    memDir->weights          = malloc(sizeof(float)*neurons*inputSize);
    memDir->bias             = malloc(sizeof(float)*neurons);
    memDir->output           = malloc(sizeof(float)*neurons);
    memDir->delta            = malloc(sizeof(float)*neurons);
    //PREALMACENAMIENTO
    memDir->weightsGrad         = malloc(sizeof(float)*neurons*inputSize);
    memDir->weightsT         = malloc(sizeof(float)*neurons*inputSize);
    memDir->wTd              = malloc(neurons*sizeof(float));
    float glorot=sqrt(6.f/(float)(inputSize+neurons));
    initMatrixRandom(memDir->neurons, memDir->inputSize, memDir->weights, -glorot, glorot);
    initMatrixRandom(memDir->neurons, 1, memDir->bias, -glorot, glorot);

}

Layer* createNetwork(int numberLayers, int neuronsLayer[numberLayers]){

    Layer *prevLayer=NULL;

    for (int i=0; i<numberLayers; i++){
        Layer *actualLayer = malloc(sizeof(Layer));    
        if (prevLayer==NULL){
            createLayer(actualLayer, i, neuronsLayer[i], 0, prevLayer);
        }else{
            createLayer(actualLayer, i, neuronsLayer[i], neuronsLayer[i-1], prevLayer);
        }
        prevLayer=actualLayer; 
    }
    Layer *pastLayer=NULL;
    Layer *actualLayer=prevLayer;
    while(actualLayer!=NULL){
        actualLayer->nextLayer=pastLayer;
        pastLayer=actualLayer;   
        actualLayer=actualLayer->prevLayer;
    }
    return prevLayer;
}

void exportNetwork(Layer *lastLayer, const char *path){
    Layer *actualLayer = lastLayer;
    char filename[128];
    while (actualLayer->prevLayer != NULL){

        snprintf(filename, sizeof(filename), "%sWEIGHT_LAYER_%d.csv", path, actualLayer->n);

        exportMatrixCsv(
            actualLayer->neurons,
            actualLayer->inputSize,
            (float (*)[actualLayer->inputSize])actualLayer->weights,
            filename
        );

        snprintf(filename, sizeof(filename), "%sBIAS_LAYER_%d.csv", path, actualLayer->n);

        exportMatrixCsv(
            actualLayer->neurons,
            1,
            (float (*)[1])actualLayer->bias,
            filename
        );

        actualLayer = actualLayer->prevLayer;
    }
}

void importNetwork(Layer *lastLayer, const char *path){
    Layer *actualLayer = lastLayer;
    char filename[128];
    while (actualLayer->prevLayer != NULL){
        snprintf(filename, sizeof(filename),"%sWEIGHT_LAYER_%d.csv", path, actualLayer->n);
        int r,c;
        importMatrixCsv(&r, &c, &actualLayer->weights, filename);
        if(r!=actualLayer->neurons || c!=actualLayer->inputSize){
            printf("ERROR IN THE WEIGHTS OF LAYER %d\n", actualLayer->n);
            printf("%d, %d\n", r,c);
            printf("%d, %d", actualLayer->neurons,actualLayer->inputSize);
            return;
        }
        snprintf(filename, sizeof(filename),"%sBIAS_LAYER_%d.csv", path, actualLayer->n);
        importMatrixCsv(&r, &c, &actualLayer->bias, filename);
        if(r!=actualLayer->neurons || c!=1){
            printf("ERROR IN THE BIAS OF LAYER %d", actualLayer->n);
            return;
        }
        actualLayer=actualLayer->prevLayer;
    }
}
void destroyNetwork(Layer *lastLayer){
    Layer *actualLayer = lastLayer;
    while (actualLayer != NULL) {
        Layer *prev = actualLayer->prevLayer;
        
        free(actualLayer->outputActivation);
        if (actualLayer->prevLayer != NULL) {
            free(actualLayer->weights);
            free(actualLayer->bias);
            free(actualLayer->output);
            free(actualLayer->delta);
            free(actualLayer->weightsT);
            free(actualLayer->wTd);
            free(actualLayer->weightsGrad);
        }
        free(actualLayer);
        actualLayer = prev;
    }
}

void printNetwork(Layer *lastLayer){
    Layer *actualLayer=lastLayer;
    while (actualLayer!=NULL){
        printf("---------------LAYER [%d]---------------\n", actualLayer->n);
        printf("LAYER Neurons:   %d\n", actualLayer->neurons);
        printf("LAYER InputSize: %d\n", actualLayer->inputSize); 
        printf("LAYER actualLayer: %p\n", actualLayer); 
        printf("LAYER nextLayer: %p\n", actualLayer->nextLayer); 
        printf("LAYER prevLayer: %p\n", actualLayer->prevLayer); 
        actualLayer=(actualLayer->prevLayer);
    }
    printf("---------------------------------------\n");
}

int importMatrixCsv(int *r, int *c, float **M1, const char *csvFile){
    FILE *f;
    char buf[64555];
    char *ptr;

    f = fopen(csvFile, "r");
    if (f == NULL) {
        printf("COULD NOT READ THE FILE: %s\n", csvFile);
        return -1;
    }
    fgets(buf, sizeof(buf), f); 
    sscanf(buf, "%d %d", r, c);
    (*M1)=malloc((*r)*(*c)*sizeof(float));
    int elements=0;
    for (int row=0; row<*r; row++){
        fgets(buf, sizeof(buf), f);
        buf[strcspn(buf, "\n")] = 0;
 
        char **arr = NULL;
        split(buf, ',', &arr);
   
        for (int col = 0; col<(*c); col++){ 
            
            (*M1)[elements]=atof(arr[col]);
            elements++;
        } 

        for(int i = 0; i < *c; i++) {
            free(arr[i]);
        }
        free(arr);
    }

    fclose(f);
    return 0;
}

int exportMatrixCsv(int r, int c, float M1[r][c], const char *csvFile){

    FILE *f;
    f=fopen(csvFile, "w");
    if (f==NULL){
        printf("COULD NOT WRITE IN THE FILE: %s\n", csvFile);
        return -1;
    }
    fprintf(f, "%d %d\n", r, c);
    for (int row = 0; row < r; row++) {
        for (int col = 0; col < c; col++) {
            fprintf(f, "%f", M1[row][col]);
            if (col!=c-1){
                fprintf(f, ",");
            }else{
                fprintf(f, "\n");
            }
        }
    }
    return 0;
}

void sigmoidMatrix(int r, int c, float M1[r][c], float result[r][c]) {
    for (int row = 0; row < r; row++) {
        for (int col = 0; col < c; col++) {
            result[row][col] = 1.0f / (1.0f + expf(-M1[row][col]));
        }
    }
}

float costFunction(int r, float M1[r][1], float expected[r][1]){
    float result=0;
    for (int row=0; row<r; row++){
        result+=powf(M1[row][0]-expected[row][0], 2);

    }
    return result/2;
}

int split (char *str, char c, char ***arr){
    int count = 1;
    int token_len = 1;
    int i = 0;
    char *p;
    char *t;

    p = str;
    while (*p != '\0')
    {
        if (*p == c)
            count++;
        p++;
    }

    *arr = (char**) malloc(sizeof(char*) * count);
    if (*arr == NULL)
        exit(1);

    p = str;
    while (*p != '\0')
    {
        if (*p == c)
        {
            (*arr)[i] = (char*) malloc( sizeof(char) * (token_len+1 ));
            if ((*arr)[i] == NULL)
                exit(1);

            token_len = 0;
            i++;
        }
        p++;
        token_len++;
    }
    (*arr)[i] = (char*) malloc( sizeof(char) * token_len );
    if ((*arr)[i] == NULL)
        exit(1);

    i = 0;
    p = str;
    t = ((*arr)[i]);
    while (*p != '\0')
    {
        if (*p != c && *p != '\0')
        {
            *t = *p;
            t++;
        }
        else
        {
            *t = '\0';
            i++;
            t = ((*arr)[i]);
        }
        p++;
    }
    *t = '\0';
    return count;
}
void sigmoidDerivateMatrix(int r, int c, float *M1, float *result){
    int totalElements=r*c;
    for (int i=0; i<totalElements; i++){
            float sig = 1.0f / (1.0f + expf(-M1[i]));
            result[i] = sig * (1.0 - sig);
        
    }
    return; 
}

float max(float a1, float a2){
    if (a1>a2) return a1;
    return a2;
}
float min(float a1, float a2){
    if (a1<a2) return a1;
    return a2;
}

