#include <stdio.h>
#include <math.h>
#include <string.h> 
#include <time.h>
#include <stdlib.h>
#include "algebra.h"

void addMatrix(int r, int c, float *M1, float *M2, float *result) {
    int totalElements = r * c;
    for (int i = 0; i < totalElements; i++) {
        result[i] = M1[i] + M2[i];
    }
}

void multMatrix(int r1, int c1, float *M1, int r2, int c2, float *M2, float *result) {
    if (c1 != r2) {
        printf("Error: Dimensión incorrecta. Cols M1 (%d) != Filas M2 (%d).\n", c1, r2);
        return;
    }

    memset(result, 0, r1 * c2 * sizeof(float));

    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            float sum = 0.0f;
            for (int k = 0; k < c1; k++) {
                sum += M1[i * c1 + k] * M2[k * c2 + j];
            }
            result[i * c2 + j] = sum;
        }
    }
}

void initMatrixRandom(int r, int c, float *M, float min, float max) {
    static int seed_initialized = 0;
    if(!seed_initialized) {
        srand(time(NULL));
        seed_initialized = 1;
    }
    
    int totalElements = r * c;
    for(int i = 0; i < totalElements; i++) {
        float scale = (float)rand() / RAND_MAX;
        M[i] = min + scale * (max - min);   
    }
}

void hadamard(int r, int c, float *M1, float *M2, float *result){
    int totalElements = r * c;
    for (int i = 0; i < totalElements; i++){
        result[i] = M1[i] * M2[i];
    }
}

void transpose(int r, int c, float *M, float *result){
    for (int i = 0; i < r; i++){
        for (int j = 0; j < c; j++){
            result[j * r + i] = M[i * c + j];
        }
    }
}

void copyMatrix(int r, int c, float *src, float *dest){
    int size = r * c * sizeof(float);
    memcpy(dest, src, size);
}

void printMatrix(int r, int c, float *M){ 
    for (int i = 0; i <= 50; i++) printf("-");
    printf("\n");
    
    for (int i = 0; i < r; i++){
        for (int j = 0; j < c; j++){
            printf("%f ", M[i * c + j]);
        }
        printf("\n");
    }
    
    for (int i = 0; i <= 50; i++) printf("-");
    printf("\n");
}

void scalarMatrixMult(int r, int c, float M[r][c], float scalar, float result[r][c]){
    for (int row = 0; row < r; row++){
        for (int col=0; col<c; col++){
            result[row][col]=M[row][col]*scalar;
        }
    }
}
