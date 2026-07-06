
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

int importMatrixCsv(int *r, int *c, float **M1, const char *csvFile);
void computeOutput(Layer *firstLayer);
int setInput(float *input, int inputSize, Layer *firstLayer);
int exportMatrixCsv(int r, int c, float M1[r][c], const char *csvFile);
int split(char *str, char c, char ***arr);
Layer* createNetwork(int numberLayers, int neuronsLayer[numberLayers]);
int readDataset(char *csvPath, int numLabels, int numFeatures, int numRow, char ***arr);
int trainNetwork(Layer *lastLayer, float *dataSet, float learningRate, int numLabels, int numFeatures, float min, float max, int datasetLines);
float testNetwork(Layer *lastLayer, float *dataSet, int numLabels, int numFeatures, float min, float max, int datasetLines);
void printNetwork(Layer *lastLayer);
void createLayer(Layer *memDir, int n, int neurons, int inputSize, Layer *prevLayer);
Layer* getFirstLayer(Layer* someLayer); // 0 first n last
Layer* getLastLayer(Layer* someLayer);
void sigmoidMatrix(int r, int c, float M1[r][c], float result[r][c]);
float costFunction(int r, float M1[r][1], float expected[r][1]);
void sigmoidDerivateMatrix(int r, int c, float *M1, float *result);
void destroyNetwork(Layer *lastLayer);
void exportNetwork(Layer *lastLayer, const char *path);
void importNetwork(Layer *lastLayer, const char *path);
float max(float a1, float a2);
float min(float a1, float a2);
#endif
