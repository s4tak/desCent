#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include "algebra.h"

void addMatrix(int r, int c, float M1[r][c], float M2[r][c], float result[r][c]) {
    for (int row = 0; row < r; row++) {
        for (int col = 0; col < c; col++) {
            result[row][col] = M1[row][col] + M2[row][col];
        }
    }
    return;
}

int exportMatrixCsv(int r, int c, float M1[r][c], const char *csvFile){
    // RETURNS -1 IF IT CANNOT WRITE A FILE

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
int importMatrixCsv(int *r, int *c, float **M1, const char *csvFile)
{
    FILE *f;
    char buf[10000];
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

void multMatrix(int r1, int c1, float M1[r1][c1], int r2, int c2, float M2[r2][c2], float result[r1][c2]) {
    if (c1 != r2) {
        printf("Error: No se pueden multiplicar matrices. Columnas de M1 deben ser iguales a filas de M2.\n");
        return;
    }
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            float sum = 0.0f;
            for (int k = 0; k < c1; k++) {
                sum += M1[i][k] * M2[k][j];
            }
            result[i][j] = sum;
        }
    }
}

void initMatrixRandom(int r, int c, float M1[r][c], float min, float max) {
    static int seed_initialized = 0;
    if(!seed_initialized) {
        srand(time(NULL));
        seed_initialized = 1;
    }

    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            float scale = (float)rand() / RAND_MAX;
            M1[i][j] = min + scale * (max - min);   
        }
    }
    return;
}
void sigmoidMatrix(int r, int c, float M1[r][c], float result[r][c]) {
    for (int row = 0; row < r; row++) {
        for (int col = 0; col < c; col++) {
            result[row][col] = 1.0f / (1.0f + expf(-M1[row][col]));
        }
    }
}

void hadamard(int r, int c, float M1[r][c], float M2[r][c], float result[r][c]){
    for (int row=0; row<r; row++){
        for (int col=0; col<c; col++){
            result[row][col]=M1[row][col]*M2[row][col];
        }
    }
    return;
}

void transpose(int r, int c, float M1[r][c], float result[c][r]){
	for (int i=0; i<r;i++){
		for (int j=0; j<c; j++){
			result[j][i]=M1[i][j];
		}
	}
	return;
}

void copyMatrix(int r, int c, float M1[r][c], float M2[r][c]){
	for (int i=0; i<r;i++){
		for (int j=0; j<c; j++){
			M2[i][j]=M1[i][j];
		}
	}
}


void printMatrix(int r, int c, float M1[r][c]){ 
    for (int i=0; i<=50; i++)
        printf("-");
    printf("\n");
    for (int i=0;i<r; i++){
        for (int j=0; j<c; j++){
            printf("%f ", M1[i][j]);
        }
        printf("\n");
    }
    for (int i=0; i<=50; i++)
        printf("-");
    printf("\n");
    return;
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
            (*arr)[i] = (char*) malloc( sizeof(char) * token_len );
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

    return count;
}

void scalarMatrixMult(int r, int c, float M1[r][c], float scalar, float result[r][c]){

    for (int row=0; row<r; row++){
        for (int col=0; col<c; col++){
            result[row][col]=scalar*M1[row][col];

        }
    }
    return;
}

void sigmoidDerivateMatrix(int r, int c, float M1[r][c], float result[r][c]){

    for (int row=0;row<r;row++){
        for (int col=0; col<c;col++){
            float sig = 1.0f / (1.0f + expf(-M1[row][col]));
            result[row][col] = sig * (1.0 - sig);
        }
    }
    return;
 
}
/*
double dot_product(int len, double v1[len], double v2[len]) {
    double result = 0;
    for (int i = 0; i < len; i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

double vector_magnitude(int len, double v[len]) {
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

void unit_vector(int len, double v[len], double result[len]) {
    double magnitude = vector_magnitude(len, v);
    if (magnitude == 0) {
        printf("Error: No se puede calcular el vector unitario de un vector nulo.\n");
        exit(1);
    }

    for (int i = 0; i < len; i++) {
        result[i] = v[i] / magnitude;
    }
}
*/
