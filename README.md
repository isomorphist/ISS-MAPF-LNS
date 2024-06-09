# ISS-MAPF-LNS

## Overview

**ISS-MAPF-LNS**: Learning to Select Promising Initial Solutions for Large Neighborhood Search-Based Multi-Agent Path Finding.

ISS-MAPF-LNS is an efficient anytime Multi-Agent Path Finding (MAPF) algorithm that leverages machine learning to select promising initial solutions in the MAPF-LNS framework.

### Key Features:
- **Integration with MAPF-LNS2**: Built on top of [MAPF-LNS2](https://github.com/Jiaoyang-Li/MAPF-LNS2) and [MAPF-LNS](https://github.com/Jiaoyang-Li/MAPF-LNS).
- **Machine Learning**: Utilizes the gradient boosting framework [LightGBM](https://github.com/microsoft/LightGBM) for training LambdaMART models.

## Documentation

### Paper
The detailed description of the algorithm can be found in the paper located in the `pub` directory.

### Experimental Evaluation
Experimental evaluation results are provided in the Jupyter Notebook available in the repository.


## Installation and Usage

The code requires the external libraries 
BOOST (https://www.boost.org/), Eigen (https://eigen.tuxfamily.org/) and LightGBM (https://github.com/microsoft/LightGBM). 
Here is an easy way of installing the required libraries on Ubuntu:
```shell script
sudo apt update
```
- Install the Eigen library (used for linear algebra computing)
    ```shell script
    sudo apt install libeigen3-dev
    ```
- Install the boost library 
    ```shell script
    sudo apt install libboost-all-dev
    ```
- Build a static-library for LightGBM and include it in CMakeLists.txt.
   
Then, go into the directory of the source code and compile it with CMake: 
```shell script
cmake -DCMAKE_BUILD_TYPE=RELEASE .
make
```

To generate training data, execute the command:
```
./lns -m MAPF-instances/mapf-map/random-32-32-20.map -a MAPF-instances/mapf-scen-random/scen-random/random-32-32-20-random-1.scen -k 400 -t 300 --neighborSize=16 --modus="generateTrainData" --pathTrainData="machine_learning/train_data" --numInitialSolutions=30 --numLNSRuns=40
```

- m: the map file from the MAPF benchmark
- a: the scenario file from the MAPF benchmark
- k: the number of agents
- t: the runtime limit
- modus: generateTrainData, testML, test
- neighborSize: the size of neighborhood
- pathTrainData: the path to training data
- numInitialSolutions: the number of initial solutions to generate
- numLNSRuns: number of LNS runs per initial solution for generating training data

Once the training data collection is completed, you can train a LambdaMART model using the JupyterNotebook located in the `machine_learning` directory.

After the model has been trained, you can use it by executing the above stated command with modus="testML" and setting parameter --pathResults to the path where your trained LambdaMART model is located.

You can find more details and explanations for all parameters with:
```
./lns --help
```

We provide all randomly generated instance sets from the [MAPF benchmark](https://movingai.com/benchmarks/mapf/index.html) in the repo folder MAPF-instances. 

The format of the scen files is explained [here](https://movingai.com/benchmarks/formats.html). 
For a given number of agents k, the first k rows of the scen file are used to generate the k pairs of start and target locations.


## Credits

The software was developed by Jiaoyang Li and Zhe Chen based on [MAPF-LNS](https://github.com/Jiaoyang-Li/MAPF-LNS) and [MAPF-LNS2](https://github.com/Jiaoyang-Li/MAPF-LNS2).

The rule-based MAPF solvers (i.e., PPS, PIBT, and winPIBT) inside the software were borrowed from 
https://github.com/Kei18/pibt/tree/v1.3

MAPF-LNS2 is released under USC – Research License. See license.txt for further details.
 
## References
[1] Jiaoyang Li, Zhe Chen, Daniel Harabor, Peter J. Stuckey and Sven Koenig.
MAPF-LNS2: Fast Repairing for Multi-Agent Path Finding via Large Neighborhood Search
In Proceedings of the AAAI Conference on Artificial Intelligence, (in print), 2022.

[2] Jiaoyang Li, Zhe Chen, Daniel Harabor, Peter J. Stuckey, Sven Koenig. 
Anytime Multi-Agent Path Finding via Large Neighborhood Search. 
In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), pages 4127-4135, 2021.
