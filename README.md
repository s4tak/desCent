# desCent

**desCent** is a lightweight C library for creating, training, saving, loading, and testing Multi-Layer Perceptron (MLP) neural networks.

## Features

- Create fully-connected MLP networks
- Train using backpropagation
- Test network performance
- Import datasets from CSV files
- Save and load trained models

---

# Training Example

```c
#include <stdio.h>
#include "ml.h"
#include "algebra.h"

int main() {
    int layers[] = {774, 16, 16, 1};
    short nLayers = sizeof(layers) / sizeof(layers[0]);

    char csvPath[] = "/path/to/dataset.csv";
    char outPath[] = "/path/to/network/";

    int rows, cols;
    float *dataset;

    Layer *network = createNetwork(nLayers, layers);

    if (importMatrixCsv(&rows, &cols, &dataset, csvPath) != 0) {
        printf("Error reading dataset.\n");
        return -1;
    }


    // Train with progressively smaller learning rates
    for (int i = 10; i <= 10000; i *= 10) {
        trainNetwork(
            network,
            dataset,
            1.0f / i,   // learning rate
            1,          // input offset
            774,        // input size
            0,          // output offset
            1,          // output size
            rows        // number of samples
        );
    }

    exportNetwork(network, outPath);

    return 0;
}
```

---

# Testing Example

```c
#include <stdio.h>
#include "ml.h"

int main() {
    int layers[] = {774, 16, 16, 1};
    short nLayers = sizeof(layers) / sizeof(layers[0]);

    char csvPath[] = "/path/to/dataset.csv";
    char outPath[] = "/path/to/network/";

    int rows, cols;
    float *dataset;

    Layer *network = createNetwork(nLayers, layers);

    if (importMatrixCsv(&rows, &cols, &dataset, csvPath) != 0) {
        printf("Error reading dataset.\n");
        return -1;
    }

    importNetwork(network, outPath);

    float cost = testNetwork(
        network,
        dataset,
        1,
        774,
        0,
        1,
        rows
    );

    printf("Average cost: %f\n", cost);

    return 0;
}
```

---

# Workflow

1. Create the network with the desired architecture.
2. Load a dataset from a CSV file.
3. Train the network using `trainNetwork()`.
4. Save the trained weights with `exportNetwork()`.
5. Load the saved network using `importNetwork()`.
6. Evaluate performance with `testNetwork()`.

---

# Network Architecture

The network architecture is defined by an array of layer sizes.

```c
int layers[] = {774, 16, 16, 1};
```

This creates:

- 774 input neurons
- 16 neurons in the first hidden layer
- 16 neurons in the second hidden layer
- 1 output neuron

---

# Dataset Format

Datasets are loaded using:

```c
importMatrixCsv(&rows, &cols, &dataset, csvPath);
```

## Dataset Format

Datasets are loaded using:

```c
importMatrixCsv(&rows, &cols, &dataset, csvPath);
```

The dataset must contain one sample per row. The first line specifies the number of rows and columns, separated by a space.

For example:

```text
507531 3
0,140,468
1,260,438
...
0,412,192
1,523,338
```

In this example:

- `507531` is the number of samples (rows).
- `3` is the number of columns.
- Each subsequent line represents one sample.
- The first value is the expected output (label).
- The remaining values are the input features.

So the row

```text
0,140,468
```

represents:

- Expected output: `0`
- Input vector: `(140, 468)`

The columns used as inputs and outputs are specified when calling `trainNetwork()` and `testNetwork()`, allowing different dataset layouts to be used without modifying the CSV format.

---

# Model Persistence

Save a trained network:

```c
exportNetwork(network, outPath);
```

Load a previously trained network:

```c
importNetwork(network, outPath);
```




