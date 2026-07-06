#ifndef ALGEBRA_H
#define ALGEBRA_H

void addMatrix(int r, int c, float *M1, float *M2, float *result);
void multMatrix(int r1, int c1, float *M1, int r2, int c2, float *M2, float *result);
void initMatrixRandom(int r, int c, float *M, float min, float max);
void hadamard(int r, int c, float *M1, float *M2, float *result);
void transpose(int r, int c, float *M, float *result);
void copyMatrix(int r, int c, float *src, float *dest);
void printMatrix(int r, int c, float *M);
void scalarMatrixMult(int r, int c, float M[r][c], float scalar, float result[r][c]);

#endif
