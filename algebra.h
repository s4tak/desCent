#ifndef BASICS_H
#define BASICS_H

void addMatrix(int r, int c, float M1[r][c], float M2[r][c], float result[r][c]);
int split(char *str, char c, char ***arr);
void multMatrix(int r1, int c1, float M1[r1][c1], int r2, int c2, float M2[r2][c2], float result[r1][c2]);
void hadamard(int r, int c, float M1[r][c], float M2[r][c], float result[r][c]);
void printMatrix(int r, int c, float M1[r][c]);
int exportMatrixCsv(int r, int c, float M1[r][c], const char *csvFile);
int importMatrixCsv(int *r, int *c, float **M1, const char *csvFile);
void copyMatrix(int r, int c, float M1[r][c], float M2[r][c]);
void transpose(int r, int c, float M1[r][c], float result[c][r]);
void scalarMatrixMult(int r, int c, float M1[r][c], float scalar, float result[r][c]);
void initMatrixRandom(int r, int c, float M1[r][c], float min, float max);
void sigmoidMatrix(int r, int c, float M1[r][c], float result[r][c]);
float costFunction(int r, float M1[r][1], float expected[r][1]);
void sigmoidDerivateMatrix(int r, int c, float M1[r][c], float result[r][c]);
#endif
