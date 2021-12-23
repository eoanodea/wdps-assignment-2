# Web Data Processing Systems - Assignment #2

## Introduction
For assignment 2, the focus shifted from word sense disambiguation to link prediction. Some solutions discussed in class were: Rule-based methods, Probabilistic methods, Factorization models, and embedding models. At first, the main focus was to explore the mechanics of link prediction behind a factorization model: RESCAL. While doing so, we could not find a well set goal for this subject. We shifted our interest towards multiple models and came across the article 'Ensemble Solutions for Link-Prediction in Knowledge Graphs' written by Denis Krompass and Volker Tresp. Krompass and Tresp proposed a solution where they ensemble multiple models to get a higher hits@k result than when one model explores a dataset. To work with different kind of models and evaluate these, the LibBKGE package is used. We epxlored four different modles; RESCAL, TransE, ComplEx, and DistMult. 

Note Alex: Good luck on the initial download, it takes like +- 20 minutes.

## Todo
- [x]  Download LibKG
- [x]  Train DBPedia (consider other models TransE, conflicts etc)
- [ ]  Link prediction using test dataset
- [ ]  Determine whether there are relations
- [ ]  Simple method to filter out noise
- [ ]  Provide high quality output
- [ ]  Which links should be kept and which should not

## Fun commands

To start the container:
```
docker-compose up
```

To find the container's name:
```
docker container ls
```

To train a model:
```
docker exec -it <container_name> /bin/sh
kge start examples/train.yaml --folder="/kge/local/experiments/main/new" --job.device cpu 
```


To train a Rescal Model
```
kge start examples/rescal.yaml --folder="/kge/local/experiments/main/rescal" --job.device cpu 
```

To clean an output file
```
python3 clean.py results/rescal/test-result.csv > output.csv
```

To see the predictions, you need to modify the KGE model located here:
```
/kge/mode/kge_model.py
```

Add this code *above* the return statement [Line 702]

```py
        tensors = self._scorer.score_emb(s, p, o, combine="sp_")
        resultList = []

        for tensorOuter in tensors:
            resultOuter = []
            for tensorInner in tensorOuter:
                resultOuter.append(tensorInner.item())
            resultList.append(resultOuter)

        print(resultList)
```

And run this commend to test

```
kge test local/experiments/main/[DIR NAME] > test-result.csv
```

## Preliminary Results

First, the 4 different models were tested and evaluated on the same dataset (i.e. DBPedia50) to see what the results of each model is. 

| Model    | Mean rank | Mean rank filtered | Mean reciprocal rank | Mean reciprocal rank filtered |
|----------|-----------|--------------------|----------------------|-------------------------------|
| RESCAL   | 360.200   | 186.395            | 0.179                | 0.354                         |
| ComplEx  | 10157.081 | 10120.005          | 0.122                | 0.132                         |
| TransE   | 6760.479  | 6723.095           | 0.095                | 0.099                         |
| DistMult | 14167.605 | 14130.465          | 0.021                | 0.021                         |




        
