
#ifndef ML_LIB
#define ML_LIB
typedef struct Layer Layer;

struct Layer{
    int n;
    int neurons;
    int inputSize;
    float *weights;
    float *bias;
    float *output;
    float *outputActivation;
    float *delta;
    Layer *prevLayer;
    Layer *nextLayer;
    float *weightsT;
    float *wTd;
    float *weightsGrad; //(dL(aL-1)T)
};

/* ==================== *
 * ======[PUBLIC]====== * 
 * ==================== */

//BASICS

Layer* createNetwork(int numberLayers, int neuronsLayer[numberLayers]);
int importMatrixCsv(int *r, int *c, float **M1, const char *csvFile); // Return 0 if all was ok
int trainNetwork(Layer *lastLayer, float *dataSet, float learningRate, int numLabels, int numFeatures, float min, float max, int datasetLines); // Return 0 if all was ok
float testNetwork(Layer *lastLayer, float *dataSet, int numLabels, int numFeatures, float min, float max, int datasetLines); //returns the avg error
void destroyNetwork(Layer *lastLayer);

// SAVE AND LOAD

void importNetwork(Layer *lastLayer, const char *path);
void exportNetwork(Layer *lastLayer, const char *path);

//DEBUG

void printNetwork(Layer *lastLayer);

/* ===================== *
 * ======[PRIVATE]====== * 
 * ===================== */


//MATH FUNCS

void sigmoidMatrix(int r, int c, float M1[r][c], float result[r][c]);
float costFunction(int r, float M1[r][1], float expected[r][1]);
void sigmoidDerivateMatrix(int r, int c, float *M1, float *result);
float max(float a1, float a2);
float min(float a1, float a2);

// Internal

void computeOutput(Layer *firstLayer);
int setInput(float *input, int inputSize, Layer *firstLayer);
int exportMatrixCsv(int r, int c, float M1[r][c], const char *csvFile);
int split(char *str, char c, char ***arr);
int readDataset(char *csvPath, int numLabels, int numFeatures, int numRow, char ***arr);
void createLayer(Layer *memDir, int n, int neurons, int inputSize, Layer *prevLayer);
Layer* getFirstLayer(Layer* someLayer); // 0 first n last
Layer* getLastLayer(Layer* someLayer);
#endif
