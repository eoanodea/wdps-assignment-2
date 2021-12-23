# Assignment 2a
Alex Antonides - 2693298 - a.m.antonides@student.vu.nl

Eoan O'Dea - 2732791 - e.odea@student.vu.nl

Tom Corten - 2618068 - t.a.corten@student.vu.nl

Max Wassenberg - 2579797 - m.n.wassenberg@student.vu.nl

## Introduction
For assignment 2, the focus shifted from word sense disambiguation to link prediction. Some solutions discussed in class were: Rule-based methods, Probabilistic methods, Factorization models, and embedding models. At first, the main focus was to explore the mechanics of link prediction behind a factorization model: RESCAL. While doing so, we could not find a well-set goal for this subject. We shifted our interest towards multiple models and came across the article 'Ensemble Solutions for Link-Prediction in Knowledge Graphs' written by Denis Krompass and Volker Tresp. Krompass and Tresp proposed a solution where they ensemble multiple models to get a higher hits@k result than when one model explores a dataset. To work with different kinds of models and evaluate these, the LibBKGE package is used. We explored four different modules; RESCAL, TransE, ComplEx, and DistMult. 

## Preliminary Results

First, the 4 different models were tested and evaluated on the same dataset (i.e. DBPedia50) to see what the results of each model is. For each of these models, the mean rank, the mean rank filtered, the mean reciprocal rank, and the mean reciprocal rank filtered were estimated as well as the unfiltered and filtered (F) hits@k.

| Model    | Mean rank | Mean rank (F) | Mean reciprocal rank | Mean reciprocal rank (F)|
|----------|-----------|---------------|----------------------|-------------------------|
| RESCAL   | 360.200   | 186.395       | 0.179                | 0.354                   |
| ComplEx  | 10157.081 | 10120.005     | 0.122                | 0.132                   |
| TransE   | 6760.479  | 6723.095      | 0.095                | 0.099                   |
| DistMult | 14167.605 | 14130.465     | 0.021                | 0.021                   |

| Model    | Hits@1 | Hits@10 | Hits@100 | Hits@1000 |
|----------|--------|---------|----------|-----------|
| RESCAL   | 0.076  | 0.138   | 0.198    | 0.281     |
| ComplEx  | 0.128  | 0.233   | 0.308    | 0.382     |
| TransE   | 0.088  | 0.202   | 0.303    | 0.400     |
| DistMult | 0.003  | 0.006   | 0.015    | 0.038     |

| Model (F)| Hits@1 | Hits@10 | Hits@100 | Hits@1000 |
|----------|--------|---------|----------|-----------|
| RESCAL   | 0.094  | 0.148   | 0.213    | 0.281     |
| ComplEx  | 0.109  | 0.177   | 0.240    | 0.313     |
| TransE   | 0.091  | 0.207   | 0.307    | 0.400     |
| DistMult | 0.003  | 0.006   | 0.015    | 0.038     |

## Results

To get to the point where we can get to ensemble multiple models from the KGE package, we scale the scores we got from the different models with the Platt Scaler.

## Technical Challenges

Various technical challenges were encountered during this project. A substantial one was extending the LIBKge library. Since the library is built around each model having a dataset and a configuration file, tuning it to take in multiple models was very challenging, and required us to replace most of the functionality so it would work on our ensemble model. Another issue was implementing the Platt scaler. It turned out to be very similar to a Linear Regression model, which only takes in a single input. This resulted in big issues for our models.

## Getting Started

These instructions will get you a copy up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
##### 1. Install the following software:
- [ ] (Option Docker)[Docker](https://www.docker.com/)

- [ ] (Option Local)[Python 3](https://www.python.corg/)
- [ ] (Option Local)[PIP](https://pip.pypa.io/en/stable/cli/pip_install/)
- [ ] (Option Local)[PyTorch](https://pytorch.org/)
  
- [ ] (Optional for Linux GPU) [CUDA](https://developer.nvidia.com/cuda-downloads)
  * See the [Supported Graphic Cards](https://developer.nvidia.com/cuda-gpus)

### Installing
A step by step series of examples that tell you how to get a development environment running.

##### 1. Update submodules
After cloning the repository, navigate to the project folder and run the following command: 
```console   
git submodule update --init --recursive
```

##### 2a. (Local) Installing the modules
If you're not using docker, navigate to the project folder and run the following command: 
```console   
python -m pip install -r requirements.txt
```

##### 2b. (Docker) Start the container
If you're using docker, run the following command:
```console   
docker-compose up [-d]
```

##### 3 Install all of the the datasets
If you're using the local option, head over to /kge/data and run "sh download_all.sh" 
If you're using the docker option, enter the container, and run "sh /kge/data/download_all.sh" 

### Development
Here are some useful tools to help you while developing!

To train a model:
```
kge start examples/train.yaml --folder="/kge/local/experiments/main/new" --job.device cpu 
```

To train a Rescal Model
```
kge start examples/rescal.yaml --folder="/kge/local/experiments/main/rescal" --job.device cpu 
```

To clean an output file
```
python3 clean_output.py results/rescal/test-result.csv > output.csv
```

To plot an output file
```
python3 plot_output.py results/rescal/test-result.csv > output.csv
```
